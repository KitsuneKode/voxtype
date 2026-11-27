// Warm microphone and queued dictation contracts. Each case runs in a private
// child process with its own runtime and data directories, so nothing here
// reads or removes the control files of a daemon running on this machine.

#[derive(Default)]
struct HeardAudio(std::sync::Mutex<Vec<Vec<f32>>>);
impl Transcriber for HeardAudio {
    fn transcribe(&self, audio: &[f32]) -> TranscriptionResult {
        self.0.lock().unwrap().push(audio.to_vec());
        Ok("heard".into())
    }
}
impl HeardAudio {
    fn takes(&self) -> Vec<Vec<f32>> {
        self.0.lock().unwrap().clone()
    }
}

/// A daemon that makes no sound, sends no notification and transcribes with
/// `HeardAudio`, publishing its state to `<root>/state`.
fn quiet_daemon(root: &Path) -> (Daemon, Arc<HeardAudio>) {
    let mut daemon = history_daemon(root);
    let n = &mut daemon.config.output.notification;
    n.on_recording_start = false;
    n.on_recording_stop = false;
    n.on_transcription = false;
    n.on_no_speech = false;
    daemon.config.hotkey.enabled = false;
    daemon.config.osd.enabled = false;
    daemon.config.tray.enabled = false;
    daemon.config.meeting.enabled = false;
    daemon.config.engine = crate::config::TranscriptionEngine::Soniox;
    let heard = Arc::new(HeardAudio::default());
    daemon.transcriber_preloaded = Some(heard.clone());
    let state = root.join("state");
    daemon.state_file_path = Some(state.clone());
    daemon.queue = Arc::new(TakeQueue::running(Some(state)));
    (daemon, heard)
}

async fn finish_transcription(daemon: &mut Daemon) {
    if let Some(task) = daemon.transcription_task.take() {
        let _ = task.await;
    }
}

/// 250 ms of pre-roll and a 60 ms tap is still a tap: it is dropped before
/// the transcriber sees it. The same tap without pre-roll, and a real 300 ms
/// press after pre-roll, are measured as before.
#[tokio::test]
async fn too_short_guard_does_not_count_pre_roll() {
    const NAME: &str = "daemon::tests::history_contracts::too_short_guard_does_not_count_pre_roll";
    if !enter_private_case(NAME) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let (mut daemon, heard) = quiet_daemon(root.path());
    let pre_roll = 4000;
    let recording = || State::Recording {
        started_at: Instant::now(),
        model_override: None,
    };

    daemon.pre_roll_samples = pre_roll;
    let mut state = recording();
    let tap = vec![0.1; pre_roll + 960];
    assert!(!daemon.transcribe_take_samples(&mut state, tap, None).await);
    assert!(state.is_idle());
    assert!(daemon.transcription_task.is_none());

    let mut state = recording();
    let press = vec![0.1; pre_roll + 4800];
    assert!(
        daemon
            .transcribe_take_samples(&mut state, press, None)
            .await
    );
    finish_transcription(&mut daemon).await;

    daemon.pre_roll_samples = 0;
    let mut state = recording();
    let cold = vec![0.1; 4960];
    assert!(daemon.transcribe_take_samples(&mut state, cold, None).await);
    finish_transcription(&mut daemon).await;

    let lengths: Vec<usize> = heard.takes().iter().map(Vec::len).collect();
    assert_eq!(
        lengths,
        [pre_roll + 4800, 4960],
        "pre-roll is still transcribed"
    );
}

/// whisper-cli is handed audio on stdin only when a take can hold pre-roll.
#[test]
fn whisper_cli_gets_stdin_audio_only_with_a_warm_mic() {
    for (warm, over_stdin) in [(0, false), (3, true)] {
        let mut config = Config::default();
        config.audio.feedback.enabled = false;
        config.state_file = None;
        config.audio.warm_mic_secs = warm;
        let daemon = Daemon::new(config, None);
        assert_eq!(
            daemon.config.whisper.audio_over_stdin, over_stdin,
            "warm_mic_secs = {warm}"
        );
    }
}

async fn until_true(mut test: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !test() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("dictation transition timed out");
}

/// With `queue_dictations` off (the default), a toggle press while the last
/// take is transcribing opens no microphone and changes no state. It waits,
/// as a key did before queueing existed, and starts the next take once the
/// first is delivered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn toggle_press_while_transcribing_waits_when_queueing_is_off() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    const NAME: &str =
        "daemon::tests::history_contracts::toggle_press_while_transcribing_waits_when_queueing_is_off";
    if !enter_private_case(NAME) {
        return;
    }
    tokio::task::LocalSet::new()
        .run_until(async {
            let root = tempfile::tempdir().unwrap();
            let mut daemon = fidelity_daemon(root.path());
            daemon.config.hotkey.enabled = false;
            daemon.config.hotkey.mode = ActivationMode::Toggle;
            daemon.config.osd.enabled = false;
            daemon.config.tray.enabled = false;
            daemon.config.meeting.enabled = false;
            daemon.config.output.file_mode = FileMode::Append;
            assert!(!daemon.config.audio.queue_dictations);
            let state_path = root.path().join("state");
            daemon.state_file_path = Some(state_path.clone());
            let (release, wait) = std::sync::mpsc::channel();
            let slow = Arc::new(LifecycleEngine {
                id: "First",
                language: "en",
                load: 0,
                fail: false,
                calls: AtomicUsize::new(0),
                completed: AtomicBool::new(false),
                gate: std::sync::Mutex::new(Some(wait)),
            });
            let heard = Arc::new(HeardAudio::default());
            let initial = State::EagerRecording {
                started_at: Instant::now(),
                model_override: None,
                accumulated_audio: vec![0.2; 2000],
                chunks_sent: 1,
                chunk_results: vec![],
                tasks_in_flight: 1,
            };
            let (events, rx) = tokio::sync::mpsc::channel(8);
            daemon.history_test_eager_loop = Some((initial, slow.clone(), heard.clone(), rx));
            daemon.history_test_capture =
                Some(Box::new(PrerecordedHistoryAudio { samples: vec![] }));
            let opened = Arc::new(AtomicBool::new(false));
            daemon
                .mic
                .fixtures
                .lock()
                .unwrap()
                .push_back(Box::new(LifecycleAudio {
                    samples: vec![0.2; 8000],
                    started: opened.clone(),
                }));
            let task = tokio::task::spawn_local(async move { daemon.run().await });
            let says = |s| std::fs::read_to_string(&state_path).as_deref().ok() == Some(s);
            let press = || HotkeyEvent::Pressed {
                model_override: None,
                profile_override: None,
            };

            until_true(|| slow.calls.load(Ordering::Acquire) > 0).await;
            events.send(press()).await.unwrap();
            until_true(|| says("transcribing")).await;

            events.send(press()).await.unwrap();
            events.send(HotkeyEvent::Released).await.unwrap();
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert!(!opened.load(Ordering::Acquire), "no second microphone");
            assert!(says("transcribing"));
            assert!(!slow.completed.load(Ordering::Acquire));

            release.send(()).unwrap();
            let entries = wait_for_entries(root.path(), 1).await;
            assert_eq!(entries.len(), 1);
            assert_eq!(
                std::fs::read_to_string(root.path().join("transcript.txt")).unwrap(),
                "First transcript\n"
            );
            until_true(|| opened.load(Ordering::Acquire) && says("recording")).await;
            events.send(HotkeyEvent::Cancel).await.unwrap();
            until_true(|| says("idle")).await;
            assert!(heard.takes().is_empty(), "nothing was queued");
            task.abort();
            let _ = task.await;
        })
        .await;
}

/// A capture whose first `prefix` samples are warm-mic pre-roll.
struct WarmAudio {
    samples: Vec<f32>,
    prefix: usize,
    started: Arc<std::sync::atomic::AtomicBool>,
}
#[async_trait::async_trait]
impl AudioCapture for WarmAudio {
    async fn start(
        &mut self,
    ) -> std::result::Result<tokio::sync::mpsc::Receiver<Vec<f32>>, crate::error::AudioError> {
        self.started
            .store(true, std::sync::atomic::Ordering::Release);
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        Ok(rx)
    }
    fn pre_roll_samples(&self) -> usize {
        self.prefix
    }
    async fn stop(&mut self) -> std::result::Result<Vec<f32>, crate::error::AudioError> {
        Ok(std::mem::take(&mut self.samples))
    }
    async fn get_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }
}

/// `pre` samples of silence followed by `pressed` samples of speech level.
fn warm_take(pre: usize, pressed: usize) -> Vec<f32> {
    let mut samples = vec![0.0; pre];
    samples.resize(pre + pressed, 0.2);
    samples
}

/// The daemon loop with a primary take held in its eager decode until
/// `release` is sent, queueing on, push-to-talk. Queued takes are
/// transcribed by `replacement`.
struct QueueHarness {
    events: tokio::sync::mpsc::Sender<HotkeyEvent>,
    release: std::sync::mpsc::Sender<()>,
    slow: Arc<LifecycleEngine>,
    state: PathBuf,
}

impl QueueHarness {
    fn new(root: &Path, replacement: Arc<dyn Transcriber>) -> (Daemon, Self) {
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        let mut daemon = fidelity_daemon(root);
        daemon.config.hotkey.enabled = false;
        daemon.config.hotkey.mode = ActivationMode::PushToTalk;
        daemon.config.audio.queue_dictations = true;
        daemon.config.osd.enabled = false;
        daemon.config.tray.enabled = false;
        daemon.config.meeting.enabled = false;
        daemon.config.output.file_mode = FileMode::Append;
        let state = root.join("state");
        daemon.state_file_path = Some(state.clone());
        let (release, wait) = std::sync::mpsc::channel();
        let slow = Arc::new(LifecycleEngine {
            id: "First",
            language: "en",
            load: 0,
            fail: false,
            calls: AtomicUsize::new(0),
            completed: AtomicBool::new(false),
            gate: std::sync::Mutex::new(Some(wait)),
        });
        let initial = State::EagerRecording {
            started_at: Instant::now(),
            model_override: None,
            accumulated_audio: vec![0.2; 2000],
            chunks_sent: 1,
            chunk_results: vec![],
            tasks_in_flight: 1,
        };
        let (events, rx) = tokio::sync::mpsc::channel(8);
        daemon.history_test_eager_loop = Some((initial, slow.clone(), replacement, rx));
        daemon.history_test_capture = Some(Box::new(PrerecordedHistoryAudio { samples: vec![] }));
        (
            daemon,
            Self {
                events,
                release,
                slow,
                state,
            },
        )
    }

    fn says(&self, state: &str) -> bool {
        std::fs::read_to_string(&self.state).as_deref().ok() == Some(state)
    }

    /// Stop the primary take; it stays transcribing until `release`.
    async fn primary_transcribing(&self) {
        until_true(|| self.slow.calls.load(std::sync::atomic::Ordering::Acquire) > 0).await;
        self.events.send(HotkeyEvent::Released).await.unwrap();
        until_true(|| self.says("transcribing")).await;
    }

    async fn press(&self, model_override: Option<&str>) {
        let event = HotkeyEvent::Pressed {
            model_override: model_override.map(str::to_string),
            profile_override: None,
        };
        self.events.send(event).await.unwrap();
    }
}

fn opened_flag() -> Arc<std::sync::atomic::AtomicBool> {
    Arc::new(std::sync::atomic::AtomicBool::new(false))
}

fn is_set(flag: &std::sync::atomic::AtomicBool) -> bool {
    flag.load(std::sync::atomic::Ordering::Acquire)
}

/// A queued start is held to what a primary start is: a model override the
/// daemon can't serve opens no microphone and says why, the start
/// notification is sent, and `--no-osd` reaches the take's OSD outcome.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_starts_check_the_model_notify_and_keep_no_osd() {
    const NAME: &str =
        "daemon::tests::history_contracts::queued_starts_check_the_model_notify_and_keep_no_osd";
    if !enter_private_case(NAME) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    let notified = root.path().join("notified");
    crate::test_support::write_script(
        &bin.join("notify-send"),
        &format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" >> {}\necho 7\n",
            notified.display()
        ),
    );
    std::env::set_var("PATH", &bin);
    let said = || std::fs::read_to_string(&notified).unwrap_or_default();
    tokio::task::LocalSet::new()
        .run_until(async {
            let heard = Arc::new(HeardAudio::default());
            let (mut daemon, h) = QueueHarness::new(root.path(), heard.clone());
            daemon.config.output.notification.on_recording_start = true;
            let opened = opened_flag();
            daemon
                .mic
                .fixtures
                .lock()
                .unwrap()
                .push_back(Box::new(LifecycleAudio {
                    samples: vec![0.2; 8000],
                    started: opened.clone(),
                }));
            let task = tokio::task::spawn_local(async move { daemon.run().await });
            h.primary_transcribing().await;

            h.press(Some("no-such-model-anywhere")).await;
            h.events.send(HotkeyEvent::Released).await.unwrap();
            until_true(|| said().contains("Model Override Failed")).await;
            tokio::time::sleep(Duration::from_millis(200)).await;
            assert!(!is_set(&opened), "a refused override opens no microphone");
            assert!(h.says("transcribing"));
            assert!(!said().contains("Recording Started"));

            let runtime = Config::runtime_dir();
            std::fs::write(runtime.join("no_osd_override"), "true").unwrap();
            h.press(None).await;
            until_true(|| is_set(&opened) && h.says("recording")).await;
            assert!(said().contains("Recording Started"));
            assert!(!runtime.join("no_osd_override").exists());
            h.events.send(HotkeyEvent::Released).await.unwrap();
            until_true(|| h.says("transcribing")).await;

            h.release.send(()).unwrap();
            assert_eq!(wait_for_entries(root.path(), 2).await.len(), 2);
            until_true(|| h.says("idle")).await;
            assert_eq!(heard.takes(), [vec![0.2; 8000]]);
            let outcome: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(runtime.join(osd::session::OUTCOME_FILE)).unwrap(),
            )
            .unwrap();
            assert_eq!(outcome["suppressed"], true, "{outcome}");
            task.abort();
            let _ = task.await;
        })
        .await;
}

/// A queued take that is cancelled, while waiting or while recording, is
/// kept for `record recover` as a primary one is, without its pre-roll.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_queued_takes_are_kept_without_pre_roll() {
    const NAME: &str =
        "daemon::tests::history_contracts::cancelled_queued_takes_are_kept_without_pre_roll";
    if !enter_private_case(NAME) {
        return;
    }
    tokio::task::LocalSet::new()
        .run_until(async {
            let root = tempfile::tempdir().unwrap();
            let heard = Arc::new(HeardAudio::default());
            let (mut daemon, h) = QueueHarness::new(root.path(), heard.clone());
            let (waiting, recording) = (opened_flag(), opened_flag());
            {
                let mut fixtures = daemon.mic.fixtures.lock().unwrap();
                fixtures.push_back(Box::new(WarmAudio {
                    samples: warm_take(4000, 48000),
                    prefix: 4000,
                    started: waiting.clone(),
                }));
                fixtures.push_back(Box::new(WarmAudio {
                    samples: warm_take(2000, 40000),
                    prefix: 2000,
                    started: recording.clone(),
                }));
            }
            let task = tokio::task::spawn_local(async move { daemon.run().await });
            let kept = crate::cancelled_take::path().unwrap();
            let recover = || {
                let samples = crate::cancelled_take::take().expect("kept for recover");
                assert!(samples.iter().all(|s| *s > 0.1), "no pre-roll silence");
                samples.len()
            };
            h.primary_transcribing().await;

            h.press(None).await;
            until_true(|| is_set(&waiting) && h.says("recording")).await;
            h.events.send(HotkeyEvent::Released).await.unwrap();
            until_true(|| h.says("transcribing")).await;
            std::fs::write(Config::runtime_dir().join("cancel"), trigger_stamp()).unwrap();
            until_true(|| kept.exists()).await;
            assert_eq!(recover(), 48000);

            h.press(None).await;
            until_true(|| is_set(&recording) && h.says("recording")).await;
            h.events.send(HotkeyEvent::Cancel).await.unwrap();
            until_true(|| kept.exists()).await;
            assert_eq!(recover(), 40000);
            until_true(|| h.says("transcribing")).await;

            h.release.send(()).unwrap();
            assert_eq!(wait_for_entries(root.path(), 1).await.len(), 1);
            until_true(|| h.says("idle")).await;
            assert!(heard.takes().is_empty());
            task.abort();
            let _ = task.await;
        })
        .await;
}

/// A capture whose start panics, as a poisoned lock or a driver bug would.
struct PanickingAudio;
#[async_trait::async_trait]
impl AudioCapture for PanickingAudio {
    async fn start(
        &mut self,
    ) -> std::result::Result<tokio::sync::mpsc::Receiver<Vec<f32>>, crate::error::AudioError> {
        panic!("capture start panicked");
    }
    async fn stop(&mut self) -> std::result::Result<Vec<f32>, crate::error::AudioError> {
        Ok(Vec::new())
    }
    async fn get_samples(&mut self) -> Vec<f32> {
        Vec::new()
    }
}

/// When the input task dies the daemon exits non-zero, so the service
/// manager restarts it, and the state file says idle until then. It used to
/// stay up with nothing reading the keys.
#[test]
fn a_dead_input_task_ends_the_daemon_with_a_failure() {
    const NAME: &str =
        "daemon::tests::history_contracts::a_dead_input_task_ends_the_daemon_with_a_failure";
    if std::env::var("VOXTYPE_HISTORY_CONTRACT_CHILD").as_deref() == Ok(NAME) {
        let root = PathBuf::from(std::env::var_os("HOME").unwrap());
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(tokio::task::LocalSet::new().run_until(async {
                let (mut daemon, h) = QueueHarness::new(&root, Arc::new(HeardAudio::default()));
                daemon
                    .mic
                    .fixtures
                    .lock()
                    .unwrap()
                    .push_back(Box::new(PanickingAudio));
                let task = tokio::task::spawn_local(async move { daemon.run().await });
                h.primary_transcribing().await;
                h.press(None).await;
                // The loop notices once it is back from the take in flight.
                tokio::time::sleep(Duration::from_millis(100)).await;
                h.release.send(()).unwrap();
                let _ = tokio::time::timeout(Duration::from_secs(5), task).await;
                panic!("the daemon stayed up without its input task");
            }));
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let runtime = root.path().join("runtime");
    std::fs::create_dir(&runtime).unwrap();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", NAME, "--nocapture"])
        .env_clear()
        .env("HOME", root.path())
        .env("TMPDIR", root.path())
        .env("XDG_RUNTIME_DIR", &runtime)
        .env("XDG_CONFIG_HOME", root.path().join("config"))
        .env("XDG_DATA_HOME", root.path().join("data"))
        .env("RUST_MIN_STACK", "16777216")
        .env("VOXTYPE_HISTORY_CONTRACT_CHILD", NAME)
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.status.code(), Some(1), "{log}");
    assert!(log.contains("capture start panicked"), "{log}");
    assert_eq!(
        std::fs::read_to_string(root.path().join("state")).unwrap(),
        "idle"
    );
}

/// At most `MAX_WAITING` stopped takes wait. A start beyond that opens no
/// microphone; once the queue drains, the next start records again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_queue_refuses_another_start() {
    const NAME: &str = "daemon::tests::history_contracts::a_full_queue_refuses_another_start";
    if !enter_private_case(NAME) {
        return;
    }
    tokio::task::LocalSet::new()
        .run_until(async {
            let root = tempfile::tempdir().unwrap();
            let heard = Arc::new(HeardAudio::default());
            let (mut daemon, h) = QueueHarness::new(root.path(), heard.clone());
            let opened: Vec<_> = (0..=take_queue::MAX_WAITING)
                .map(|_| opened_flag())
                .collect();
            for flag in &opened {
                daemon
                    .mic
                    .fixtures
                    .lock()
                    .unwrap()
                    .push_back(Box::new(LifecycleAudio {
                        samples: vec![0.2; 8000],
                        started: flag.clone(),
                    }));
            }
            let task = tokio::task::spawn_local(async move { daemon.run().await });
            h.primary_transcribing().await;
            for flag in &opened[..take_queue::MAX_WAITING] {
                h.press(None).await;
                until_true(|| is_set(flag) && h.says("recording")).await;
                h.events.send(HotkeyEvent::Released).await.unwrap();
                until_true(|| h.says("transcribing")).await;
            }

            let refused = &opened[take_queue::MAX_WAITING];
            h.press(None).await;
            tokio::time::sleep(Duration::from_millis(200)).await;
            assert!(!is_set(refused), "a full queue opens no microphone");
            assert!(h.says("transcribing"));
            h.events.send(HotkeyEvent::Released).await.unwrap();

            h.release.send(()).unwrap();
            let delivered = 1 + take_queue::MAX_WAITING;
            assert_eq!(
                wait_for_entries(root.path(), delivered).await.len(),
                delivered
            );
            until_true(|| h.says("idle")).await;
            assert_eq!(heard.takes().len(), take_queue::MAX_WAITING);
            h.press(None).await;
            until_true(|| is_set(refused) && h.says("recording")).await;
            h.events.send(HotkeyEvent::Cancel).await.unwrap();
            until_true(|| h.says("idle")).await;
            task.abort();
            let _ = task.await;
        })
        .await;
}

/// `record recover` settles its options when it starts, as a stop does, so a
/// queued take that writes its own while the recovered one transcribes
/// neither takes them nor gives it its own.
#[tokio::test]
async fn a_recovered_take_keeps_its_own_options() {
    const NAME: &str = "daemon::tests::history_contracts::a_recovered_take_keeps_its_own_options";
    if !enter_private_case(NAME) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let (mut daemon, heard) = quiet_daemon(root.path());
    let runtime = Config::runtime_dir();
    std::fs::create_dir_all(&runtime).unwrap();
    assert!(crate::cancelled_take::keep(&vec![0.2; 48000]));
    let recovered = root.path().join("recovered.txt");
    let output = runtime.join("output_mode_override");
    std::fs::write(&output, format!("file:{}", recovered.display())).unwrap();

    let mut state = State::Idle;
    daemon.recover_cancelled_take(&mut state).await;
    assert!(matches!(state, State::Transcribing { .. }));
    assert_eq!(
        daemon.frozen.as_ref().and_then(|o| o.output.clone()),
        Some(OutputOverride::FileWithPath(recovered.clone()))
    );
    assert!(!output.exists(), "read when the recovery started");

    std::fs::write(&output, "clipboard").unwrap();
    let result = daemon.transcription_task.take().unwrap().await;
    daemon.handle_transcription_result(&mut state, result).await;
    assert_eq!(heard.takes().len(), 1);
    assert!(std::fs::read_to_string(&recovered)
        .unwrap()
        .to_lowercase()
        .contains("heard"));
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "clipboard");
}

/// At exit, media still ducked comes back at once rather than over the
/// configured fade, which the 2 s exit budget could cut short; a restore
/// already fading up when the daemon stops is waited for, not aborted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn media_is_restored_without_a_fade_on_exit() {
    const NAME: &str = "daemon::tests::history_contracts::media_is_restored_without_a_fade_on_exit";
    if !enter_private_case(NAME) {
        return;
    }
    use crate::audio::media::DuckedMediaStream;
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    let log = root.path().join("pactl");
    crate::test_support::write_script(
        &bin.join("pactl"),
        &format!("#!/bin/sh\necho \"$*\" >> {}\n", log.display()),
    );
    std::env::set_var("PATH", &bin);
    let volume_writes = || -> Vec<String> {
        let all = std::fs::read_to_string(&log).unwrap_or_default();
        let _ = std::fs::remove_file(&log);
        all.lines()
            .filter(|l| l.starts_with("set-sink-input-volume"))
            .map(str::to_string)
            .collect()
    };
    let mut config = crate::config::AudioConfig::default();
    config.duck_media = true;
    config.duck_media_volume_percent = 30;
    config.duck_media_fade_ms = 1500;
    let media = recording_media::Media::default();

    let lease = media
        .ducked(
            &config,
            vec![DuckedMediaStream::recorded(42, &["30000", "30000"])],
        )
        .await;
    let started = Instant::now();
    media.restore_before_exit(Some(lease)).await;
    assert!(started.elapsed() < Duration::from_millis(1000));
    assert_eq!(volume_writes(), ["set-sink-input-volume 42 30000 30000"]);

    config.duck_media_fade_ms = 300;
    let lease = media
        .ducked(&config, vec![DuckedMediaStream::recorded(7, &["20000"])])
        .await;
    drop(lease);
    media.restore_before_exit(None).await;
    let writes = volume_writes();
    assert!(writes.len() > 1, "the fade ran: {writes:?}");
    assert_eq!(writes.last().unwrap(), "set-sink-input-volume 7 20000");
}
