//! A stopped take owns the delivery pipeline until its output finishes. The
//! input task can keep recording in the meantime. Only the daemon takes work
//! from the front, so neither a live streaming driver nor a slow output hook
//! can interleave two dictations.
use super::*;
use std::collections::VecDeque;
use std::sync::Mutex;
use tokio::sync::{mpsc, oneshot, watch, Notify};

#[derive(Default)]
pub(super) struct Overrides {
    pub output: Option<OutputOverride>,
    pub profile: Option<String>,
    pub post_process: Option<bool>,
    pub smart_submit: Option<bool>,
    pub auto_submit: Option<bool>,
    pub shift_enter: Option<bool>,
}

impl Overrides {
    pub fn read() -> Self {
        Self {
            output: read_output_mode_override(),
            profile: read_profile_override(),
            post_process: read_bool_override("post_process"),
            smart_submit: read_bool_override("smart_auto_submit"),
            auto_submit: read_bool_override("auto_submit"),
            shift_enter: read_bool_override("shift_enter"),
        }
    }
}

pub(super) struct Take {
    pub sequence: u64,
    pub model: Option<String>,
    pub options: Overrides,
    pub samples: Vec<f32>,
    pub prefix: usize,
    pub started: Instant,
    pub external: bool,
}

#[derive(Default)]
struct Order {
    sequence: u64,
    primary: &'static str,
    processing: bool,
    cancel: bool,
    skip: bool,
    recording: bool,
    waiting: VecDeque<Take>,
}

impl Order {
    fn state(&self) -> &'static str {
        if self.recording {
            "recording"
        } else if self.processing || !self.waiting.is_empty() {
            "transcribing"
        } else {
            self.primary
        }
    }
    fn next(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }
}

#[derive(Default)]
pub(super) struct TakeQueue {
    order: Mutex<Order>,
    path: Option<PathBuf>,
    /// Where `record` leaves its control files (`cancel`, `cancel_ack`,
    /// `start_requested`): the runtime directory, or a test's own.
    dir: PathBuf,
    managed: bool,
    processing: watch::Sender<bool>,
    pub ready: Notify,
}

impl TakeQueue {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            dir: Config::runtime_dir(),
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub fn running_in(path: Option<PathBuf>, dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            ..Self::running(path)
        }
    }

    pub fn running(path: Option<PathBuf>) -> Self {
        Self {
            managed: true,
            ..Self::new(path)
        }
    }
    fn change<R>(&self, change: impl FnOnce(&mut Order) -> R) -> R {
        let mut order = self.order.lock().unwrap();
        let before = order.state();
        let result = change(&mut order);
        if order.state() != before {
            if let Some(path) = &self.path {
                write_state_file(path, order.state());
            }
        }
        result
    }
    pub fn primary(&self, state: &'static str) {
        self.change(|q| {
            q.primary = state;
            if state == "idle" {
                q.processing = false;
                q.cancel = false;
                q.skip = false;
            }
            if self.managed && state == "transcribing" {
                q.processing = true;
            }
        });
        self.processing.send_replace(state == "transcribing");
    }
    pub fn primary_finished_capture(&self) {
        let sequence = self.change(|q| q.next());
        tracing::debug!(sequence, "Dictation capture finished");
        self.primary("transcribing");
    }
    fn accepting_overlap(&self) -> bool {
        let q = self.order.lock().unwrap();
        q.processing || q.primary == "transcribing" || q.recording || !q.waiting.is_empty()
    }
    pub fn skip_post_process(&self) -> bool {
        let mut q = self.order.lock().unwrap();
        if std::mem::take(&mut q.skip) {
            return true;
        }
        !q.recording
            && q.waiting.is_empty()
            && !self.dir.join(START_REQUEST_FILE).exists()
            && read_bool_override("post_process") == Some(false)
    }
    pub fn cancel_primary(&self) -> bool {
        let mut q = self.order.lock().unwrap();
        if std::mem::take(&mut q.cancel) {
            return true;
        }
        if q.recording || !q.waiting.is_empty() {
            return false;
        }
        let cancel = self.dir.join("cancel");
        if cancel_targets_start(
            read_trigger_stamp(&self.dir.join(START_REQUEST_FILE)),
            read_trigger_stamp(&cancel),
        ) {
            return false;
        }
        std::fs::remove_file(cancel).is_ok()
    }
    pub fn next_for_delivery(&self) -> Option<Take> {
        self.change(|q| {
            if q.primary != "idle" {
                return None;
            }
            let take = q.waiting.pop_front()?;
            q.primary = "transcribing";
            q.processing = self.managed;
            Some(take)
        })
    }

    fn finish(&self, mut take: Take) {
        self.change(|q| {
            take.sequence = q.next();
            q.waiting.push_back(take);
            q.recording = false;
        });
        self.ready.notify_one();
    }
    fn cancel_waiting(&self) -> Option<Take> {
        self.change(|q| {
            let take = q.waiting.pop_back()?;
            acknowledge_queue_cancel(&self.dir);
            Some(take)
        })
    }
}

pub(super) enum Input {
    Key(HotkeyEvent),
    Start,
    Stop,
}
pub(super) struct Routed(pub Input, pub oneshot::Sender<()>);

struct Recording {
    capture: Box<dyn AudioCapture>,
    model: Option<String>,
    profile: Option<String>,
    started: Instant,
    external: bool,
    tracker: Option<audio::levels::SpeechTracker>,
    media: Option<recording_media::Lease>,
}

pub(super) struct CaptureServices {
    pub mic: audio::cpal_capture::DictationMic,
    pub levels: Option<audio::levels::LevelHub>,
    pub media: recording_media::Media,
    pub feedback: Option<audio::feedback::Player>,
}
impl CaptureServices {
    fn play(&self, event: SoundEvent) {
        if let Some(f) = &self.feedback {
            f.play(event);
        }
    }
}

pub(super) struct Inputs {
    pub keys: Option<mpsc::Receiver<HotkeyEvent>>,
    pub start: tokio::signal::unix::Signal,
    pub stop: tokio::signal::unix::Signal,
}

impl Inputs {
    async fn recv(&mut self) -> Input {
        tokio::select! {
            Some(key) = async {
                match &mut self.keys { Some(rx) => rx.recv().await, None => std::future::pending().await }
            } => Input::Key(key),
            _ = self.start.recv() => Input::Start,
            _ = self.stop.recv() => Input::Stop,
        }
    }

    pub async fn route(
        mut self,
        primary: mpsc::Sender<Routed>,
        queue: Arc<TakeQueue>,
        config: Config,
        services: CaptureServices,
    ) {
        // Off, every input goes to the daemon's loop as it did before
        // queueing existed, and this task never opens the microphone.
        let queueing = config.audio.queue_dictations;
        let mut active: Option<Recording> = None;
        let mut tick = tokio::time::interval(Duration::from_millis(50));
        let mut processing = queue.processing.subscribe();
        loop {
            let input = tokio::select! {
                biased;
                _ = primary.closed() => return,
                input = self.recv() => input,
                _ = tick.tick(), if queueing && queue.accepting_overlap() => {
                    if (active.is_some() || !queue.order.lock().unwrap().waiting.is_empty()) && queue.dir.join("cancel").exists() {
                        Input::Key(HotkeyEvent::Cancel)
                    } else if let Some(r) = &active {
                        let silent = match (&r.tracker, config.audio.external_trigger_silence_timeout_secs) {
                            (Some(t), Some(limit)) => t.silence_elapsed().await.as_secs_f32() >= limit,
                            _ => false,
                        };
                        if silent || r.started.elapsed().as_secs() >= u64::from(config.audio.max_duration_secs) { Input::Stop } else { continue }
                    } else {continue}
                }
            };
            let _stop_ack = if matches!(input, Input::Stop) {
                Some(StopAck)
            } else {
                None
            };
            if !queueing
                || !queue.accepting_overlap()
                || matches!(&input, Input::Start) && queue.dir.join("recover").exists()
            {
                let (ack, done) = oneshot::channel();
                processing.borrow_and_update();
                if primary.send(Routed(input, ack)).await.is_err() {
                    return;
                }
                // A stop can itself await the eager decode or final pass.
                // Resume intake as soon as it has relinquished the microphone.
                tokio::select! {
                    _ = done => {},
                    _ = processing.wait_for(|busy| *busy) => {},
                }
                continue;
            }
            let cancelled = matches!(input, Input::Key(HotkeyEvent::Cancel));
            let start = match input {
                Input::Key(HotkeyEvent::Cancel) => {
                    if active.is_none() {
                        if let Some(take) = queue.cancel_waiting() {
                            report_cancelled(&config, &take.options);
                        } else {
                            queue.change(|q| q.cancel = true);
                        }
                        continue;
                    }
                    None
                }
                Input::Key(HotkeyEvent::PasteLast) => continue,
                Input::Key(HotkeyEvent::Released)
                    if config.hotkey.mode == ActivationMode::Toggle =>
                {
                    continue
                }
                Input::Key(HotkeyEvent::Pressed {
                    model_override,
                    profile_override,
                }) => {
                    if active.is_none() {
                        Some((model_override, profile_override, false))
                    } else if config.hotkey.mode == ActivationMode::PushToTalk {
                        continue;
                    } else {
                        None
                    }
                }
                Input::Start if active.is_none() => Some((None, None, true)),
                Input::Stop if active.is_none() => {
                    let options = Overrides::read();
                    queue.change(|q| {
                        if let Some(take) = q.waiting.back_mut() {
                            if let Some(v) = options.post_process {
                                take.options.post_process = Some(v);
                            }
                        } else if options.post_process == Some(false) {
                            q.skip = true;
                        }
                    });
                    continue;
                }
                Input::Start => {
                    drop(StartRequest::read());
                    continue;
                }
                Input::Key(HotkeyEvent::Released) | Input::Stop => None,
            };
            if let Some((model, profile, external)) = start {
                let request = external.then(StartRequest::read);
                let older_cancel = queue.change(|q| {
                    let cancel = queue.dir.join("cancel");
                    let mut dropped = None;
                    if cancel.exists()
                        && !cancel_targets_start(
                            request.as_ref().and_then(|r| r.issued_at),
                            read_trigger_stamp(&cancel),
                        )
                    {
                        dropped = q.waiting.pop_back();
                        if dropped.is_none() {
                            q.cancel = true;
                        }
                        acknowledge_queue_cancel(&queue.dir);
                    }
                    q.recording = true;
                    dropped
                });
                if let Some(take) = older_cancel {
                    report_cancelled(&config, &take.options);
                }
                if external {
                    apply_start_options();
                }
                let model = model.or_else(read_model_override);
                let _ = read_bool_override("no_osd");
                let result = async {
                    let media = services.media.acquire(&config.audio).await;
                    let mut capture = services.mic.capture(&config.audio)?;
                    let chunks = capture.start().await?;
                    let tracker = external
                        .then_some(config.audio.external_trigger_silence_timeout_secs)
                        .flatten()
                        .map(|_| {
                            audio::levels::SpeechTracker::new(
                                config.audio.external_trigger_speech_threshold_dbfs,
                            )
                        });
                    if let Some(hub) = &services.levels {
                        audio::levels::spawn_emitter_with_streaming_tap(
                            chunks,
                            hub.frame_sink(),
                            None,
                            None,
                            tracker.clone(),
                        );
                    } else if let Some(t) = tracker.clone() {
                        audio::levels::spawn_silence_tracker(chunks, t);
                    }
                    Ok::<_, crate::error::AudioError>(Recording {
                        capture,
                        model,
                        profile,
                        external,
                        started: Instant::now(),
                        tracker,
                        media,
                    })
                }
                .await;
                match result {
                    Ok(r) => {
                        active = Some(r);
                        services.play(SoundEvent::RecordingStart);
                        if let Some(cmd) = &config.output.pre_recording_command {
                            if let Err(e) = output::run_hook(cmd, "pre_recording").await {
                                tracing::warn!("{e}");
                            }
                        }
                    }
                    Err(e) => {
                        let options = Overrides::read();
                        report_outcome(&config, &options, TranscriptOutcome::error(&e.to_string()));
                        services.play(SoundEvent::Error);
                        tracing::error!("Queued microphone capture failed: {e}");
                        queue.change(|q| q.recording = false);
                    }
                }
            } else if let Some(mut r) = active.take() {
                let prefix = r.capture.pre_roll_samples();
                let samples = r.capture.stop().await;
                r.media.take();
                let mut options = Overrides::read();
                options.profile = options.profile.or(r.profile);
                if cancelled || queue.dir.join("cancel").exists() {
                    services.play(SoundEvent::Cancelled);
                    report_cancelled(&config, &options);
                    queue.change(|q| {
                        acknowledge_queue_cancel(&queue.dir);
                        q.recording = false;
                    });
                } else if let Ok(samples) = samples {
                    services.play(SoundEvent::RecordingStop);
                    queue.finish(Take {
                        sequence: 0,
                        samples,
                        prefix,
                        options,
                        started: r.started,
                        external: r.external,
                        model: r.model,
                    });
                } else {
                    report_cancelled(&config, &options);
                    queue.change(|q| q.recording = false);
                }
                if r.external {
                    if let Some(cmd) = &config.audio.external_trigger_stop_command {
                        let _ = output::run_hook(cmd, "external_trigger_stop").await;
                    }
                }
            }
        }
    }
}

fn report_cancelled(config: &Config, options: &Overrides) {
    report_outcome(config, options, TranscriptOutcome::empty());
}
fn report_outcome(config: &Config, options: &Overrides, outcome: TranscriptOutcome) {
    let out = apply_output_override(&config.output, options.output.clone());
    if out.mode == OutputMode::File {
        if let Some(path) = out.file_path {
            write_result_sidecar(&path, &outcome);
        }
    }
}

fn acknowledge_queue_cancel(dir: &Path) {
    if let Some(stamp) = read_trigger_stamp(&dir.join("cancel")) {
        let _ = std::fs::write(dir.join("cancel_ack"), stamp.to_string());
    }
    let _ = std::fs::remove_file(dir.join("cancel"));
}

pub(super) struct ControlFiles;
impl ControlFiles {
    pub fn publish() -> std::io::Result<Self> {
        let dir = Config::runtime_dir();
        for key in ["stop_requested", "start_options", "cancel_ack"] {
            let _ = std::fs::remove_file(dir.join(key));
        }
        std::fs::write(dir.join("queue_pid"), std::process::id().to_string())?;
        Ok(Self)
    }
}
impl Drop for ControlFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(Config::runtime_dir().join("queue_pid"));
    }
}
struct StopAck;
impl Drop for StopAck {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(Config::runtime_dir().join("stop_requested"));
    }
}
pub(super) fn apply_start_options() {
    let dir = Config::runtime_dir();
    let path = dir.join("start_options");
    if let Ok(bytes) = std::fs::read(&path) {
        let _ = std::fs::remove_file(path);
        if let Ok(options) =
            serde_json::from_slice::<std::collections::BTreeMap<String, String>>(&bytes)
        {
            for (key, value) in options {
                if RECORDING_OVERRIDE_FILES.contains(&key.as_str()) || key == "no_osd_override" {
                    if let Err(e) = std::fs::write(dir.join(key), value) {
                        tracing::warn!("Recording option: {e}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_take(tag: f32) -> Take {
        Take {
            sequence: 0,
            model: None,
            options: Overrides::default(),
            samples: vec![tag; 160],
            prefix: 0,
            started: Instant::now(),
            external: false,
        }
    }

    #[test]
    fn sequence_numbers_and_state_file_stay_in_order_across_overlapping_takes() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let queue = TakeQueue::running_in(Some(state.clone()), dir.path());
        let read_state = || std::fs::read_to_string(&state).unwrap();

        queue.primary("idle");
        assert_eq!(read_state(), "idle");

        // Primary starts and stops recording (sequence 1).
        queue.primary("recording");
        assert_eq!(read_state(), "recording");
        queue.primary_finished_capture();
        assert_eq!(read_state(), "transcribing");
        assert!(queue.accepting_overlap());

        // Overlapping take 2 records and finishes while take 1 is transcribing.
        queue.change(|q| q.recording = true);
        assert_eq!(read_state(), "recording");
        queue.finish(sample_take(2.0));
        assert_eq!(read_state(), "transcribing");

        // Overlapping take 3 records and finishes while take 1 is still transcribing.
        queue.change(|q| q.recording = true);
        assert_eq!(read_state(), "recording");
        queue.finish(sample_take(3.0));
        assert_eq!(read_state(), "transcribing");

        // While take 1 is still transcribing, delivery does not release take 2.
        assert!(queue.next_for_delivery().is_none());

        // Take 1 finishes: state must stay "transcribing" (no idle flicker) because take 2 & 3 wait.
        queue.primary("idle");
        assert_eq!(read_state(), "transcribing");

        // Take 2 is popped next with sequence 2, then take 3 with sequence 3.
        let t2 = queue.next_for_delivery().expect("take 2 ready");
        assert_eq!(t2.sequence, 2);
        assert_eq!(t2.samples[0], 2.0);
        assert_eq!(read_state(), "transcribing");
        assert!(queue.next_for_delivery().is_none());

        // Cancel waiting take 3 before take 2 finishes: take 3 is dropped.
        let dropped = queue.cancel_waiting().expect("take 3 dropped");
        assert_eq!(dropped.sequence, 3);

        // Take 2 finishes with nothing left waiting or recording: state becomes idle immediately.
        queue.primary("idle");
        assert_eq!(read_state(), "idle");
        assert!(!queue.accepting_overlap());
        assert!(queue.next_for_delivery().is_none());
    }

    #[test]
    fn active_overlapping_recording_is_not_overwritten_when_primary_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let queue = TakeQueue::running_in(Some(state.clone()), dir.path());
        let read_state = || std::fs::read_to_string(&state).unwrap();

        queue.primary_finished_capture();
        assert_eq!(read_state(), "transcribing");

        queue.change(|q| q.recording = true);
        assert_eq!(read_state(), "recording");

        // Primary finishes while the new take is still recording.
        queue.primary("idle");
        assert_eq!(read_state(), "recording");
        assert!(queue.accepting_overlap());

        // Cancelling the active recording with no other work returns straight to idle.
        queue.change(|q| q.recording = false);
        assert_eq!(read_state(), "idle");
        assert!(!queue.accepting_overlap());
    }
}
