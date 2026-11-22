//! Final pass that keeps up with the recording.
//!
//! The streaming final pass (`[parakeet] streaming_final_pass`) used to wait
//! for the stop and then decode the whole recording, so the wait at stop grew
//! with the recording: about 66 ms per second of audio on a laptop CPU, or
//! 20 s after five minutes of dictation. With long recordings split at
//! pauses anyway (`segment::plan_segments`), every piece that is complete
//! while the user is still speaking can be decoded right then. At stop only
//! the unfinished last piece is left, so the wait is bounded by one piece
//! however long the recording is.
//!
//! The tap sits between the audio capture and the level/streaming pump and
//! keeps its own copy of the samples. The pump forwards the same chunks in
//! the same order, and the capture's `stop()` returns that same sample
//! stream, so an offset into the copy is an offset into the final
//! recording (the drained-stop path relies on the same property).

use super::segment::plan_segments;
use super::Transcriber;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// How often the worker looks for a finished piece.
const POLL: Duration = Duration::from_millis(500);

/// A piece is cut only once the uncommitted audio is this much longer than
/// the piece limit, so the planner always has the whole candidate window
/// [limit/2, limit] plus some audio after it to judge the pause.
const LOOKAHEAD_SECS: f32 = 1.0;

/// What the worker decoded before the stop.
#[derive(Debug, Default)]
struct Progress {
    /// Samples (from the start of the recording) covered by `parts`.
    committed: usize,
    /// Text of each committed piece, in order.
    parts: Vec<String>,
}

pub struct IncrementalFinalPass {
    /// Flipped to `true` at stop/cancel; wakes the worker immediately.
    stop: tokio::sync::watch::Sender<bool>,
    worker: tokio::task::JoinHandle<Progress>,
    transcriber: Arc<dyn Transcriber>,
}

impl IncrementalFinalPass {
    /// Start tapping `chunk_rx`. Returns the worker and the receiver the
    /// caller should read instead of `chunk_rx` (same chunks, same order).
    pub fn start(
        mut chunk_rx: mpsc::Receiver<Vec<f32>>,
        transcriber: Arc<dyn Transcriber>,
        sample_rate: u32,
        segment_secs: f32,
    ) -> (Self, mpsc::Receiver<Vec<f32>>) {
        let recorded = Arc::new(Mutex::new(Vec::<f32>::new()));
        let (fwd_tx, fwd_rx) = mpsc::channel::<Vec<f32>>(64);

        let rec = recorded.clone();
        tokio::spawn(async move {
            let mut forwarding = true;
            while let Some(chunk) = chunk_rx.recv().await {
                if let Ok(mut r) = rec.lock() {
                    r.extend_from_slice(&chunk);
                }
                // The pump goes away on stop/cancel; keep recording until the
                // capture closes so the copy stays a prefix of the recording.
                if forwarding && fwd_tx.send(chunk).await.is_err() {
                    forwarding = false;
                }
            }
        });

        let (stop, mut stop_rx) = tokio::sync::watch::channel(false);
        let t = transcriber.clone();
        let max_len = (segment_secs * sample_rate as f32) as usize;
        let trigger = ((segment_secs + LOOKAHEAD_SECS) * sample_rate as f32) as usize;
        let worker = tokio::spawn(async move {
            let mut progress = Progress::default();
            loop {
                // Wake on the stop at once: the stop is waiting on us.
                tokio::select! {
                    _ = tokio::time::sleep(POLL) => {}
                    _ = stop_rx.changed() => break,
                }
                if *stop_rx.borrow() {
                    break;
                }
                let tail: Vec<f32> = match recorded.lock() {
                    Ok(r) if r.len() >= progress.committed + trigger => r[progress.committed..].to_vec(),
                    _ => continue,
                };
                let (_, end) = plan_segments(&tail, sample_rate, segment_secs)[0];
                let end = end.min(max_len).min(tail.len());
                let piece = tail[..end].to_vec();
                let started = std::time::Instant::now();
                let t2 = t.clone();
                match tokio::task::spawn_blocking(move || t2.transcribe_piece(&piece)).await {
                    Ok(Ok(text)) => {
                        tracing::info!(
                            "Final pass: decoded {:.1}s piece in the background in {}ms",
                            end as f32 / sample_rate as f32,
                            started.elapsed().as_millis()
                        );
                        let text = text.trim().to_string();
                        if !text.is_empty() {
                            progress.parts.push(text);
                        }
                        progress.committed += end;
                    }
                    Ok(Err(e)) => {
                        // Leave the rest to the decode at stop.
                        tracing::warn!("Background final-pass piece failed: {}", e);
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("Background final-pass task failed: {}", e);
                        break;
                    }
                }
            }
            progress
        });

        (
            Self {
                stop,
                worker,
                transcriber,
            },
            fwd_rx,
        )
    }

    /// Finish the final pass for the whole recording `captured`: stop the
    /// worker (letting a piece it is decoding finish, since that is work the
    /// stop would otherwise redo), then decode what it had not reached.
    pub async fn finish(self, captured: Arc<Vec<f32>>) -> Result<String, crate::error::TranscribeError> {
        let _ = self.stop.send(true);
        let mut progress = self.worker.await.unwrap_or_default();
        if progress.committed > captured.len() {
            // Should not happen (the copy is a prefix of the recording), but
            // never stitch text onto audio that does not line up.
            tracing::warn!("Final pass: background pieces do not match the recording; decoding it whole");
            progress = Progress::default();
        }
        let pieces = progress.parts.len();
        let tail = captured[progress.committed..].to_vec();
        let t = self.transcriber.clone();
        let tail_text = if tail.is_empty() {
            String::new()
        } else {
            tokio::task::spawn_blocking(move || t.transcribe(&tail))
                .await
                .map_err(|e| crate::error::TranscribeError::InferenceFailed(e.to_string()))??
        };
        let mut parts = progress.parts;
        let tail_text = tail_text.trim();
        if !tail_text.is_empty() {
            parts.push(tail_text.to_string());
        }
        if pieces > 0 {
            tracing::debug!("Final pass: {} background piece(s) + tail", pieces);
        }
        Ok(parts.join(" "))
    }

    /// Drop the session's work (cancel). A piece mid-decode finishes in the
    /// background and is discarded.
    pub fn abandon(self) {
        let _ = self.stop.send(true);
        self.worker.abort();
    }
}
