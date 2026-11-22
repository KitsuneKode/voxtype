//! Streaming transcription support
//!
//! Defines the [`StreamingTranscriber`] trait and supporting event types
//! for backends that emit transcribed text incrementally as audio arrives,
//! rather than after the user releases the hotkey.
//!
//! ## Why a separate trait
//!
//! The classic [`Transcriber`](super::Transcriber) trait is a one-shot
//! operation: take the recorded buffer, return the final text. Streaming
//! providers (Gemini Live, OpenAI Realtime, ElevenLabs Scribe v2, the
//! whisper.cpp `stream` example, etc.) instead push partial and final
//! segments over a long-lived connection.
//!
//! Rather than retrofit every existing backend, streaming-capable backends
//! opt in by implementing this trait *in addition* to `Transcriber`. The
//! daemon discovers streaming capability via
//! [`Transcriber::as_streaming`](super::Transcriber::as_streaming).
//!
//! ## Lifecycle
//!
//! ```text
//!     daemon                              backend
//!       │                                    │
//!       │  start_stream(samples_rx)          │
//!       │ ─────────────────────────────────▶ │
//!       │                                    │ (open WS / start worker)
//!       │  StreamHandle{events, cancel}      │
//!       │ ◀───────────────────────────────── │
//!       │                                    │
//!       │  push samples on samples_rx        │
//!       │ ─────────────────────────────────▶ │
//!       │                                    │
//!       │            Partial / Final         │
//!       │ ◀───────────────────────────────── │
//!       │            ...                     │
//!       │                                    │
//!       │  drop samples_rx (graceful end)    │
//!       │ ─────────────────────────────────▶ │
//!       │                                    │ (flush, close)
//!       │              Ended                 │
//!       │ ◀───────────────────────────────── │
//! ```
//!
//! The daemon may also send on `cancel` to abort early. After cancel,
//! backends should stop emitting events as soon as practical.

use crate::error::TranscribeError;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Identifier for a logical segment within a streaming session.
///
/// Backends are free to define what constitutes a segment — for VAD-driven
/// providers it usually corresponds to one utterance between pauses; for
/// fixed-window providers it may be one chunk. The daemon treats segments
/// as opaque grouping for *Partial* → *Final* progression.
pub type SegmentId = u64;

/// Events emitted by a streaming transcription backend.
///
/// Not `Clone`: the `Error` variant carries [`TranscribeError`], which is
/// not clonable. Events flow one-to-one over an mpsc channel, so cloning
/// is not needed in normal use.
#[derive(Debug)]
pub enum StreamingEvent {
    /// In-progress text for a segment. May be revised by later partials
    /// or superseded by a `Final` event with the same `segment_id`.
    Partial { text: String, segment_id: SegmentId },

    /// Committed text for a segment. The daemon's default output policy
    /// is to type only `Final` segments, so revision-style providers do
    /// not produce visible churn.
    Final { text: String, segment_id: SegmentId },

    /// Backspace `backspace` chars then commit `text`. Used by streaming
    /// backends (notably Soniox) that revise the tail of a previously
    /// typed non-final tail when finalizing — e.g. punctuation flips
    /// from `,` to `.` or a token spelling changes between non-final
    /// and final. Equivalent to `Final` when `backspace == 0`.
    Replace {
        backspace: usize,
        text: String,
        segment_id: SegmentId,
    },

    /// The backend has finished processing all audio sent so far and is
    /// closing the stream gracefully (e.g., the daemon dropped the
    /// samples sender and the backend has flushed).
    Ended,

    /// A non-recoverable error occurred. The daemon should treat this
    /// like `Ended` for state purposes and surface the error to the user.
    Error(TranscribeError),
}

/// Handle returned by [`StreamingTranscriber::start_stream`].
///
/// The daemon polls `events` and may signal early termination via
/// `cancel`. `task` is held so the daemon can `await` shutdown.
pub struct StreamHandle {
    /// Events from the backend (partials, finals, end, errors).
    pub events: mpsc::Receiver<StreamingEvent>,

    /// Send `()` to abort the stream early. Dropping this sender is a
    /// no-op; backends should also treat the samples sender being
    /// dropped as a graceful end-of-input.
    pub cancel: oneshot::Sender<()>,

    /// The background task driving the backend connection. Awaiting this
    /// after `events` produces `Ended` ensures resources are reclaimed.
    pub task: JoinHandle<Result<(), TranscribeError>>,
}

/// How far the current stream has got, for backends that track it.
/// Sample positions count from the start of the stream's audio, which is
/// also the start of the recording the daemon captures.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamProgress {
    /// Everything the stream has emitted so far.
    pub text: String,
    /// Samples handed to the model.
    pub fed: usize,
    /// Where the audio behind `text` ends: the end of the last chunk that
    /// produced text.
    pub text_through: usize,
}

/// `[parakeet] streaming_final_pass_skip_when_clean`: whether the streamed
/// text can stand in for the final pass. It must end a sentence (`.` `?`
/// `!` `…`), and at least 0.3 s of the recording must follow the audio
/// behind it, all of it quiet (every 20 ms frame closer to the noise floor
/// than to the speech level, the relative test `segment` cuts at).
/// Anything the stream had not decoded yet lies in that stretch, so a
/// stream that fell behind never passes.
pub fn stream_is_clean(progress: &StreamProgress, audio: &[f32], sample_rate: u32) -> bool {
    if !progress.text.trim_end().ends_with(['.', '?', '!', '…']) {
        return false;
    }
    let frame = (sample_rate as usize / 50).max(1);
    let tail = &audio[progress.text_through.min(audio.len())..];
    if tail.len() < sample_rate as usize * 3 / 10 {
        return false;
    }
    let db =
        |c: &[f32]| 10.0 * (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32 + 1e-10).log10();
    let mut levels: Vec<f32> = audio.chunks_exact(frame).map(db).collect();
    levels.sort_by(f32::total_cmp);
    let floor = levels[levels.len() / 10];
    let speech = levels[levels.len() * 9 / 10];
    let threshold = floor + 0.25 * (speech - floor);
    tail.chunks_exact(frame).all(|c| db(c) < threshold)
}

/// Trait for transcription backends that can stream partial results.
///
/// Implementors should:
///
/// 1. Accept 16 kHz mono `f32` samples on `samples_rx` (matching
///    [`crate::audio::AudioCapture`]'s output).
/// 2. Emit at least one `Final` per committed utterance, plus zero or
///    more `Partial`s leading up to it.
/// 3. Emit `Ended` exactly once when the stream closes cleanly. If the
///    `cancel` signal fires, backends may emit `Ended` immediately
///    without flushing pending partials.
/// 4. Treat the `samples_rx` sender being dropped as graceful EOF.
///
/// Errors during setup (auth, network, model load) should be returned
/// from `start_stream` directly. Errors during the session should arrive
/// as `StreamingEvent::Error` followed by `Ended`.
pub trait StreamingTranscriber: Send + Sync {
    /// Open a streaming session.
    ///
    /// `samples_rx` is the channel from which the backend will pull
    /// audio. The daemon will keep this sender alive for as long as the
    /// user is recording, then drop it to signal end-of-input.
    fn start_stream(
        &self,
        samples_rx: mpsc::Receiver<Vec<f32>>,
    ) -> Result<StreamHandle, TranscribeError>;

    /// Progress of the most recent stream, for backends that track it.
    fn stream_progress(&self) -> Option<StreamProgress> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_sentence_ends_followed_by_quiet_audio_are_clean() {
        let speech = |i: usize| (i as f32 * 0.05).sin() * 0.2;
        let noise = |i: usize| ((i * 7919) % 97) as f32 * 2e-5;
        // 2 s of speech, then 0.6 s of room noise.
        let audio: Vec<f32> = (0..41600)
            .map(|i| {
                if i < 32000 {
                    speech(i) + noise(i)
                } else {
                    noise(i)
                }
            })
            .collect();
        let at = |text: &str, through: usize| StreamProgress {
            text: text.into(),
            fed: audio.len(),
            text_through: through,
        };
        assert!(stream_is_clean(&at("Done.", 32000), &audio, 16000));
        assert!(stream_is_clean(&at("Really?", 33000), &audio, 16000));
        // No sentence end: the speaker may have more to say.
        assert!(!stream_is_clean(&at("Done", 32000), &audio, 16000));
        // The stream's text ends inside the speech: it is behind.
        assert!(!stream_is_clean(&at("Done.", 20000), &audio, 16000));
        // Too little audio after the text to tell.
        assert!(!stream_is_clean(&at("Done.", 38000), &audio, 16000));
    }

    #[test]
    fn streaming_event_partial_fields() {
        let ev = StreamingEvent::Partial {
            text: "hello".into(),
            segment_id: 1,
        };
        match ev {
            StreamingEvent::Partial { text, segment_id } => {
                assert_eq!(text, "hello");
                assert_eq!(segment_id, 1);
            }
            _ => panic!("expected Partial"),
        }
    }

    #[test]
    fn streaming_event_final_distinct_from_partial() {
        let p = StreamingEvent::Partial {
            text: "x".into(),
            segment_id: 1,
        };
        let f = StreamingEvent::Final {
            text: "x".into(),
            segment_id: 1,
        };
        // Sanity: discriminant difference matters for the daemon's match arms.
        assert!(matches!(p, StreamingEvent::Partial { .. }));
        assert!(matches!(f, StreamingEvent::Final { .. }));
    }
}
