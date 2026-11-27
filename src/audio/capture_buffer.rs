//! Only an active take owns a resampler and a streaming tap. Between takes
//! keep mono device-rate samples so even a resampler delay line cannot hold
//! more ambient sound than the configured pre-roll.

use super::resampler::{resample_buffer, StreamResampler};
use std::collections::VecDeque;
use tokio::sync::mpsc;

pub(super) enum CaptureBuffer {
    Recording {
        samples: Vec<f32>,
        tx: mpsc::Sender<Vec<f32>>,
        resampler: Option<Box<StreamResampler>>,
    },
    Warm {
        ring: VecDeque<f32>,
        limit: usize,
    },
    Closed,
}

impl CaptureBuffer {
    pub(super) fn start(&mut self, tx: mpsc::Sender<Vec<f32>>, source: u32, target: u32) -> usize {
        let prefix = match self {
            Self::Warm { ring, .. } => resample_buffer(ring.make_contiguous(), source, target),
            _ => Vec::new(),
        };
        let len = prefix.len();
        if len > 0 {
            let _ = tx.try_send(prefix.clone());
        }
        let resampler = (source != target)
            .then(|| {
                StreamResampler::new(source, target)
                    .inspect_err(|e| {
                        tracing::error!("{e}; capture will run without rate conversion")
                    })
                    .ok()
            })
            .flatten()
            .map(Box::new);
        *self = Self::Recording {
            samples: prefix,
            tx,
            resampler,
        };
        len
    }

    pub(super) fn push(&mut self, mono: Vec<f32>) {
        match self {
            Self::Recording {
                samples,
                tx,
                resampler,
            } => {
                let chunk = match resampler {
                    Some(r) => r.push(&mono),
                    None => mono,
                };
                samples.extend_from_slice(&chunk);
                let _ = tx.try_send(chunk);
            }
            Self::Warm { ring, limit } => {
                let tail = &mono[mono.len().saturating_sub(*limit)..];
                ring.drain(..ring.len().saturating_sub(*limit - tail.len()));
                ring.extend(tail);
            }
            Self::Closed => {}
        }
    }

    pub(super) fn drain(&mut self) -> Vec<f32> {
        match self {
            Self::Recording { samples, .. } => std::mem::take(samples),
            _ => Vec::new(),
        }
    }

    pub(super) fn stop(&mut self, warm_limit: Option<usize>) -> Vec<f32> {
        let next = warm_limit.map_or(Self::Closed, |limit| Self::Warm {
            ring: VecDeque::new(),
            limit,
        });
        match std::mem::replace(self, next) {
            Self::Recording {
                mut samples,
                mut resampler,
                ..
            } => {
                if let Some(r) = resampler.as_mut() {
                    samples.extend(r.flush());
                }
                samples
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warm_ring_is_bounded_and_feeds_batch_and_streaming_once() {
        let mut b = CaptureBuffer::Closed;
        b.stop(Some(3));
        for chunk in [vec![1., 2.], vec![3., 4.], vec![5., 6., 7., 8.], vec![9.]] {
            b.push(chunk);
        }
        let (tx, mut rx) = mpsc::channel(4);
        assert_eq!(b.start(tx, 16000, 16000), 3);
        b.push(vec![10.]);
        assert_eq!(rx.try_recv().unwrap(), [7., 8., 9.]);
        assert_eq!(rx.try_recv().unwrap(), [10.]);
        assert_eq!(b.stop(Some(3)), [7., 8., 9., 10.]);
        b.push(vec![11.]);
        b.stop(None);
        let (tx, _) = mpsc::channel(1);
        assert_eq!(b.start(tx, 16000, 16000), 0);
    }

    /// Without a warm window a take is exactly its chunks, in the buffer
    /// and on the streaming channel alike, and nothing outlives its stop.
    #[test]
    fn a_cold_take_is_exactly_what_was_captured() {
        let mut b = CaptureBuffer::Closed;
        b.push(vec![9., 9.]);
        let (tx, mut rx) = mpsc::channel(4);
        assert_eq!(b.start(tx, 16000, 16000), 0);
        b.push(vec![1., 2.]);
        b.push(vec![3.]);
        assert_eq!(rx.try_recv().unwrap(), [1., 2.]);
        assert_eq!(rx.try_recv().unwrap(), [3.]);
        assert!(rx.try_recv().is_err());
        assert_eq!(b.stop(None), [1., 2., 3.]);
        b.push(vec![4.]);
        assert!(matches!(b, CaptureBuffer::Closed));
    }

    #[test]
    fn warm_ring_with_zero_pre_roll_retains_nothing() {
        let mut b = CaptureBuffer::Closed;
        b.stop(Some(0));
        b.push(vec![1., 2., 3.]);
        let (tx, mut rx) = mpsc::channel(4);
        assert_eq!(b.start(tx, 16000, 16000), 0);
        b.push(vec![4., 5.]);
        assert_eq!(rx.try_recv().unwrap(), [4., 5.]);
        assert_eq!(b.stop(None), [4., 5.]);
    }
}
