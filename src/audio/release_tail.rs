//! The end-of-take buffer (`[audio] release_tail_ms`).
//!
//! A stop request arrives while the last syllable may still be in the air,
//! so a dictation stop keeps the microphone capturing for a short tail whose
//! length follows the audio: speech still running at the stop keeps the tail
//! going until a short run of silence or the configured cap, and a take that
//! is already silent ends after a small guard. Cancel never takes a tail.

use super::AudioCapture;
use crate::config::VadConfig;
use crate::vad::{frame_rms, speech_level};
use std::time::Duration;
use tokio::time::Instant;

const SAMPLE_RATE: usize = 16_000;
/// Level frames of 10 ms, the OSD meter's rate.
const FRAME: usize = SAMPLE_RATE / 100;
/// The audio just before the stop that decides whether speech is running.
const LOOKBACK_FRAMES: usize = 10;
/// Loud frames out of the lookback that mean speech: 60 ms of the last
/// 100 ms. A key's click is one or two frames, not six.
const SPEAKING_FRAMES: usize = 6;
/// How long a take that is already silent keeps listening.
pub const GUARD: Duration = Duration::from_millis(70);
/// The run of silence that ends a tail started on speech.
pub const HANGOVER: Duration = Duration::from_millis(150);
/// Audio before the stop the speech level is measured against.
pub const HISTORY_SAMPLES: usize = 3 * SAMPLE_RATE;
const POLL: Duration = Duration::from_millis(10);
/// Wall-clock allowance past the tail's own length, for a device that stops
/// delivering samples.
const STALL: Duration = Duration::from_millis(200);

fn samples_in(d: Duration) -> usize {
    (d.as_micros() as usize).saturating_mul(SAMPLE_RATE) / 1_000_000
}

/// Decides, from the audio itself, when a release tail has heard enough.
pub struct TailDecider {
    level: f32,
    speaking: bool,
    limit: usize,
    hangover: usize,
    heard: usize,
    quiet: usize,
    frame: Vec<f32>,
}

impl TailDecider {
    /// `before` is the take up to the stop (its most recent
    /// [`HISTORY_SAMPLES`] are enough); `cap` is the longest tail allowed.
    pub fn new(before: &[f32], cap: Duration, vad: &VadConfig) -> Self {
        let aligned = &before[before.len() % FRAME..];
        let frames: Vec<f32> = aligned
            .as_chunks::<FRAME>()
            .0
            .iter()
            .map(|frame| frame_rms(frame))
            .collect();
        let level = speech_level(vad, &frames);
        let loud = frames
            .iter()
            .rev()
            .take(LOOKBACK_FRAMES)
            .filter(|&&rms| rms >= level)
            .count();
        let speaking = loud >= SPEAKING_FRAMES;
        let cap = samples_in(cap);
        Self {
            level,
            speaking,
            limit: if speaking {
                cap
            } else {
                samples_in(GUARD).min(cap)
            },
            hangover: samples_in(HANGOVER),
            heard: 0,
            quiet: 0,
            frame: Vec::with_capacity(FRAME),
        }
    }

    /// Whether speech was running when the stop arrived.
    pub fn speaking(&self) -> bool {
        self.speaking
    }

    /// The longest this tail can run.
    pub fn budget(&self) -> Duration {
        Duration::from_micros((self.limit * 1_000_000 / SAMPLE_RATE) as u64)
    }

    pub fn done(&self) -> bool {
        self.heard >= self.limit || (self.speaking && self.quiet >= self.hangover)
    }

    /// Feed audio captured after the stop; returns [`Self::done`].
    pub fn push(&mut self, chunk: &[f32]) -> bool {
        for &sample in chunk {
            if self.done() {
                break;
            }
            self.frame.push(sample);
            if self.frame.len() == FRAME {
                self.heard += FRAME;
                if frame_rms(&self.frame) >= self.level {
                    self.quiet = 0;
                } else {
                    self.quiet += FRAME;
                }
                self.frame.clear();
            }
        }
        self.done()
    }
}

/// Keep `capture` recording after a stop request for the tail its audio
/// calls for, at most `cap`, and return how many samples the tail added.
/// Returns at once when `cap` is zero, when `interrupted` (a cancel) is
/// true, or when the capture cannot be read while it records.
pub async fn hold(
    capture: &mut dyn AudioCapture,
    cap: Duration,
    vad: &VadConfig,
    interrupted: impl Fn() -> bool,
) -> usize {
    if cap.is_zero() || interrupted() {
        return 0;
    }
    let Some((start, before)) = capture.captured_since(0, HISTORY_SAMPLES).await else {
        return 0;
    };
    let mut decider = TailDecider::new(&before, cap, vad);
    let deadline = Instant::now() + decider.budget() + STALL;
    let mut seen = start;
    while !decider.done() && Instant::now() < deadline && !interrupted() {
        tokio::time::sleep(POLL).await;
        let Some((total, fresh)) = capture.captured_since(seen, usize::MAX).await else {
            break;
        };
        seen = seen.max(total);
        decider.push(&fresh);
    }
    let added = seen - start;
    tracing::debug!(
        "Release tail: {} ms ({})",
        added * 1000 / SAMPLE_RATE,
        if decider.speaking() {
            "speech at the stop"
        } else {
            "silent at the stop"
        }
    );
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AudioError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc;

    const MS: usize = SAMPLE_RATE / 1000;

    fn tone(ms: usize, amplitude: f32) -> Vec<f32> {
        (0..ms * MS)
            .map(|i| amplitude * (i as f32 * 0.07).sin())
            .collect()
    }

    fn room(ms: usize) -> Vec<f32> {
        tone(ms, 0.0005)
    }

    fn speech(ms: usize) -> Vec<f32> {
        tone(ms, 0.2)
    }

    fn cat(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.concat()
    }

    /// A take that has captured `before` and goes on to capture `after`,
    /// 10 ms per read, then room tone for as long as it is read.
    struct Scripted {
        captured: Vec<f32>,
        after: Vec<f32>,
        reads: Arc<AtomicUsize>,
    }

    impl Scripted {
        fn new(before: Vec<f32>, after: Vec<f32>) -> Self {
            Self {
                captured: before,
                after,
                reads: Arc::default(),
            }
        }
    }

    #[async_trait::async_trait]
    impl AudioCapture for Scripted {
        async fn start(&mut self) -> Result<mpsc::Receiver<Vec<f32>>, AudioError> {
            Ok(mpsc::channel(1).1)
        }
        async fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
            Ok(std::mem::take(&mut self.captured))
        }
        async fn get_samples(&mut self) -> Vec<f32> {
            Vec::new()
        }
        async fn captured_since(
            &mut self,
            from: usize,
            at_most: usize,
        ) -> Option<(usize, Vec<f32>)> {
            if self.reads.fetch_add(1, Ordering::Relaxed) > 0 {
                let n = FRAME.min(self.after.len());
                let next: Vec<f32> = self.after.drain(..n).collect();
                self.captured
                    .extend(if next.is_empty() { room(10) } else { next });
            }
            let total = self.captured.len();
            let from = from.max(total.saturating_sub(at_most)).min(total);
            Some((total, self.captured[from..].to_vec()))
        }
    }

    fn vad() -> VadConfig {
        VadConfig::default()
    }

    #[test]
    fn speech_at_the_stop_runs_until_a_run_of_silence() {
        let before = cat(&[room(1000), speech(800)]);
        let mut d = TailDecider::new(&before, Duration::from_millis(1000), &vad());
        assert!(d.speaking());
        assert!(!d.push(&speech(200)));
        assert!(!d.push(&room(140)));
        assert!(d.push(&room(10)), "150 ms of silence ends it");
        assert_eq!(d.heard, 350 * MS);
    }

    #[test]
    fn a_word_resuming_in_the_tail_restarts_the_silence_run() {
        let before = cat(&[room(1000), speech(500)]);
        let mut d = TailDecider::new(&before, Duration::from_millis(1000), &vad());
        assert!(!d.push(&cat(&[room(100), speech(60), room(140)])));
        assert!(d.push(&room(10)));
        assert_eq!(d.heard, 310 * MS);
    }

    #[test]
    fn silence_at_the_stop_ends_after_the_guard() {
        let before = cat(&[room(500), speech(600), room(300)]);
        let mut d = TailDecider::new(&before, Duration::from_millis(1000), &vad());
        assert!(!d.speaking());
        assert_eq!(d.budget(), GUARD);
        assert!(!d.push(&room(60)));
        assert!(d.push(&room(10)));
        assert_eq!(d.heard, 70 * MS);
    }

    #[test]
    fn a_key_click_at_the_stop_is_not_speech() {
        let before = cat(&[room(1000), speech(20), room(80)]);
        let d = TailDecider::new(&before, Duration::from_millis(1000), &vad());
        assert!(!d.speaking());
    }

    #[test]
    fn the_cap_holds_while_speech_goes_on() {
        let before = cat(&[room(500), speech(500)]);
        let mut d = TailDecider::new(&before, Duration::from_millis(400), &vad());
        assert!(!d.push(&speech(390)));
        assert!(d.push(&speech(1000)));
        assert_eq!(d.heard, 400 * MS);
        let quiet = cat(&[room(500), speech(300), room(200)]);
        let d = TailDecider::new(&quiet, Duration::from_millis(30), &vad());
        assert_eq!(
            d.budget(),
            Duration::from_millis(30),
            "a cap below the guard wins"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn hold_extends_a_take_still_speaking_until_silence() {
        let mut take = Scripted::new(cat(&[room(1000), speech(700)]), speech(200));
        let added = hold(&mut take, Duration::from_millis(1000), &vad(), || false).await;
        assert_eq!(added, 350 * MS);
        assert_eq!(take.stop().await.unwrap().len(), (1700 + 350) * MS);
    }

    #[tokio::test(start_paused = true)]
    async fn hold_stops_a_silent_take_after_the_guard() {
        let mut take = Scripted::new(cat(&[room(1000), speech(700), room(400)]), vec![]);
        let added = hold(&mut take, Duration::from_millis(1000), &vad(), || false).await;
        assert_eq!(added, 70 * MS);
    }

    #[tokio::test(start_paused = true)]
    async fn hold_never_runs_past_the_cap() {
        let mut take = Scripted::new(cat(&[room(300), speech(700)]), speech(5000));
        let added = hold(&mut take, Duration::from_millis(500), &vad(), || false).await;
        assert_eq!(added, 500 * MS);
    }

    #[tokio::test(start_paused = true)]
    async fn off_or_cancelled_reads_nothing_and_adds_nothing() {
        for (cap, cancelled) in [(0, false), (1000, true)] {
            let mut take = Scripted::new(cat(&[room(300), speech(700)]), speech(500));
            let reads = take.reads.clone();
            let added = hold(&mut take, Duration::from_millis(cap), &vad(), || cancelled).await;
            assert_eq!(added, 0);
            assert_eq!(reads.load(Ordering::Relaxed), 0);
            assert_eq!(take.stop().await.unwrap().len(), 1000 * MS);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_cancel_during_the_tail_ends_it() {
        let mut take = Scripted::new(cat(&[room(300), speech(700)]), speech(5000));
        let reads = take.reads.clone();
        let added = hold(&mut take, Duration::from_millis(1000), &vad(), || {
            reads.load(Ordering::Relaxed) > 5
        })
        .await;
        assert_eq!(added, 50 * MS);
    }

    /// A capture that can't be read while it records (a test fixture, a
    /// future backend) is stopped as it always was.
    #[tokio::test(start_paused = true)]
    async fn a_capture_without_a_live_read_takes_no_tail() {
        struct Opaque;
        #[async_trait::async_trait]
        impl AudioCapture for Opaque {
            async fn start(&mut self) -> Result<mpsc::Receiver<Vec<f32>>, AudioError> {
                Ok(mpsc::channel(1).1)
            }
            async fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
                Ok(vec![])
            }
            async fn get_samples(&mut self) -> Vec<f32> {
                vec![]
            }
        }
        let started = tokio::time::Instant::now();
        assert_eq!(
            hold(&mut Opaque, Duration::from_millis(1000), &vad(), || false).await,
            0
        );
        assert_eq!(started.elapsed(), Duration::ZERO);
    }
}
