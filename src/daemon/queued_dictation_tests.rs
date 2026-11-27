/// Hold the primary eager decode across a second press/release. The next
/// engine already has its answer, but only the daemon may deliver it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_take_delivery_and_cancel() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct ReadyTranscript(std::sync::Mutex<std::sync::mpsc::Receiver<String>>);
    impl Transcriber for ReadyTranscript {
        fn transcribe(&self, audio: &[f32]) -> TranscriptionResult {
            assert_eq!(audio, vec![0.2; 8000]);
            Ok(self
                .0
                .lock()
                .unwrap()
                .try_recv()
                .expect("the later decode completed before the first"))
        }
    }
    const NAME: &str = "daemon::tests::history_contracts::queued_take_delivery_and_cancel";
    if !enter_private_case(NAME) {
        return;
    }
    tokio::task::LocalSet::new()
        .run_until(async {
            let root = tempfile::tempdir().unwrap();
            let mut daemon = fidelity_daemon(root.path());
            daemon.config.hotkey.enabled = false;
            daemon.config.hotkey.mode = ActivationMode::PushToTalk;
            daemon.config.audio.queue_dictations = true;
            daemon.config.osd.enabled = false;
            daemon.config.tray.enabled = false;
            daemon.config.meeting.enabled = false;
            daemon.config.output.file_mode = FileMode::Append;
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
            let fast = Arc::new(LifecycleEngine {
                id: "Second",
                language: "en",
                load: 0,
                fail: false,
                calls: AtomicUsize::new(0),
                completed: AtomicBool::new(false),
                gate: Default::default(),
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
            let (finished, result) = std::sync::mpsc::channel();
            daemon.history_test_eager_loop = Some((
                initial,
                slow.clone(),
                Arc::new(ReadyTranscript(result.into())),
                rx,
            ));
            daemon.history_test_capture =
                Some(Box::new(PrerecordedHistoryAudio { samples: vec![] }));
            let mut captures = Vec::new();
            for _ in 0..3 {
                let started = Arc::new(AtomicBool::new(false));
                daemon
                    .mic
                    .fixtures
                    .lock()
                    .unwrap()
                    .push_back(Box::new(LifecycleAudio {
                        samples: vec![0.2; 8000],
                        started: started.clone(),
                    }));
                captures.push(started);
            }
            let task = tokio::task::spawn_local(async move { daemon.run().await });
            async fn until(mut test: impl FnMut() -> bool) {
                tokio::time::timeout(Duration::from_secs(5), async {
                    while !test() {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                })
                .await
                .expect("dictation transition timed out");
            }
            let says = |s| std::fs::read_to_string(&state_path).as_deref().ok() == Some(s);
            until(|| slow.calls.load(Ordering::Acquire) > 0).await;
            events.send(HotkeyEvent::Released).await.unwrap();
            until(|| says("transcribing")).await;

            for started in &captures[..2] {
                events
                    .send(HotkeyEvent::Pressed {
                        model_override: None,
                        profile_override: None,
                    })
                    .await
                    .unwrap();
                until(|| started.load(Ordering::Acquire) && says("recording")).await;
                events.send(HotkeyEvent::Released).await.unwrap();
                until(|| says("transcribing")).await;
            }

            // Drop the most recently stopped take, with the earlier one still waiting.
            let runtime = Config::runtime_dir();
            let stamp = trigger_stamp();
            std::fs::write(runtime.join("cancel"), &stamp).unwrap();
            until(|| {
                std::fs::read_to_string(runtime.join("cancel_ack")).ok() == Some(stamp.clone())
            })
            .await;
            assert!(!slow.completed.load(Ordering::Acquire));
            assert!(!root.path().join("transcript.txt").exists());

            // A third press starts before the earlier takes deliver. They must
            // not publish idle or transcribing over this active microphone.
            events
                .send(HotkeyEvent::Pressed {
                    model_override: None,
                    profile_override: None,
                })
                .await
                .unwrap();
            until(|| captures[2].load(Ordering::Acquire) && says("recording")).await;
            finished
                .send(fast.transcribe(&vec![0.2; 8000]).unwrap())
                .unwrap();
            assert!(!slow.completed.load(Ordering::Acquire));
            release.send(()).unwrap();
            let entries = wait_for_entries(root.path(), 2).await;
            assert_eq!(entries.len(), 2);
            assert_eq!(fast.calls.load(Ordering::Acquire), 1);
            assert!(says("recording"));
            events.send(HotkeyEvent::Cancel).await.unwrap();
            until(|| says("idle")).await;
            assert_eq!(
                std::fs::read_to_string(root.path().join("transcript.txt")).unwrap(),
                "First transcript\nSecond transcript\n"
            );
            task.abort();
            let _ = task.await;
        })
        .await;
}
