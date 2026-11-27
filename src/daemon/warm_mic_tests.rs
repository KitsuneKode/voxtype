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
