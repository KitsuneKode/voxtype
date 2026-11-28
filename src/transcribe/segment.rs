//! Split a long recording at its pauses for an offline decode.
//!
//! Full-context models (Parakeet's offline pass) get slower than linear with
//! length (attention) and, past a minute or so, start dropping punctuation
//! and casing. Decoding each stretch between pauses on its own keeps every
//! word's context to the sentence or two around it, which is what the model
//! was trained on, and keeps the cost linear.
//!
//! The planner only cuts inside silence: it looks for the longest quiet run
//! in the back half of each window and cuts at its middle, so both sides keep
//! a little silence and no word is split. If a window has no quiet frame at
//! all (continuous speech for `max_secs`), it cuts at the quietest frame.

use super::Transcriber;
use crate::error::TranscribeError;

/// Decode `audio` in pieces of at most `piece_secs`, cut at pauses (see
/// [`plan_segments`]), the last one as the end of the recording. A piece
/// needs a fraction of the GPU memory the whole recording does, so this
/// still decodes when another process has taken the memory a whole-take
/// decode needs to grow into, where decoding the whole take again fails the
/// same way.
pub fn transcribe_in_pieces(
    transcriber: &dyn Transcriber,
    audio: &[f32],
    sample_rate: u32,
    piece_secs: f32,
) -> Result<String, TranscribeError> {
    let mut parts = Vec::new();
    for (start, end) in plan_segments(audio, sample_rate, piece_secs) {
        let piece = &audio[start..end];
        let text = if end == audio.len() {
            transcriber.transcribe(piece)?
        } else {
            transcriber.transcribe_piece(piece)?
        };
        let text = text.trim();
        if !text.is_empty() {
            parts.push(text.to_string());
        }
    }
    Ok(parts.join(" "))
}

/// Analysis frame: 20 ms.
const FRAME_SECS: f32 = 0.02;

/// Plan `(start, end)` sample ranges covering `audio`, each at most
/// `max_secs` long (and, unless the recording is shorter, at least half of
/// that), cut at pauses. A recording that fits in one segment comes back as
/// a single range, so callers can treat the short case and the long case the
/// same way.
pub fn plan_segments(audio: &[f32], sample_rate: u32, max_secs: f32) -> Vec<(usize, usize)> {
    let len = audio.len();
    let max_len = (max_secs.max(1.0) * sample_rate as f32) as usize;
    if len <= max_len {
        return vec![(0, len)];
    }

    let frame = ((FRAME_SECS * sample_rate as f32) as usize).max(1);
    let db: Vec<f32> = audio
        .chunks(frame)
        .map(|c| {
            let e = c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32;
            10.0 * (e + 1e-10).log10()
        })
        .collect();

    // Quiet = closer to the noise floor than to the speech level. Relative
    // levels, so mic gain and room noise do not matter.
    let mut sorted = db.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[sorted.len() / 10];
    let speech = sorted[sorted.len() * 9 / 10];
    let threshold = floor + 0.25 * (speech - floor);

    let max_frames = max_len / frame;
    let min_frames = max_frames / 2;
    let mut cuts = Vec::new();
    let mut start = 0usize; // in frames
    while db.len() - start > max_frames {
        let lo = start + min_frames;
        let hi = (start + max_frames).min(db.len());
        // Longest quiet run inside [lo, hi); ties go to the later run so
        // segments stay long.
        let (mut best, mut best_len) = (None, 0usize);
        let mut run_start = None;
        for i in lo..=hi {
            let quiet = i < hi && db[i] < threshold;
            match (quiet, run_start) {
                (true, None) => run_start = Some(i),
                (false, Some(s)) => {
                    if i - s >= best_len {
                        best_len = i - s;
                        best = Some((s + i) / 2);
                    }
                    run_start = None;
                }
                _ => {}
            }
        }
        let cut = best.unwrap_or_else(|| {
            (lo..hi)
                .min_by(|&a, &b| {
                    db[a]
                        .partial_cmp(&db[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(hi)
        });
        cuts.push(cut);
        start = cut;
    }

    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut s = 0usize;
    for c in cuts {
        let e = (c * frame).min(len);
        out.push((s, e));
        s = e;
    }
    out.push((s, len));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32) -> Vec<f32> {
        (0..(secs * 16000.0) as usize)
            .map(|i| 0.3 * (i as f32 * 0.07).sin())
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * 16000.0) as usize]
    }

    #[test]
    fn short_audio_is_one_segment() {
        let a = tone(10.0);
        assert_eq!(plan_segments(&a, 16000, 30.0), vec![(0, a.len())]);
    }

    #[test]
    fn cuts_land_in_pauses_and_cover_everything() {
        // 4 x (12 s speech + 0.5 s pause) = 50 s.
        let mut a = Vec::new();
        let mut pauses = Vec::new();
        for _ in 0..4 {
            a.extend(tone(12.0));
            let p0 = a.len();
            a.extend(silence(0.5));
            pauses.push(p0..a.len());
        }
        let segs = plan_segments(&a, 16000, 30.0);
        assert!(segs.len() >= 2);
        assert_eq!(segs[0].0, 0);
        assert_eq!(segs.last().unwrap().1, a.len());
        for w in segs.windows(2) {
            assert_eq!(w[0].1, w[1].0, "segments must be contiguous");
            assert!(
                pauses.iter().any(|p| p.contains(&w[0].1)),
                "cut at {} is not inside a pause",
                w[0].1
            );
        }
        for (s, e) in &segs {
            assert!(e - s <= 30 * 16000);
        }
    }

    #[test]
    fn continuous_speech_still_respects_max() {
        let a = tone(70.0);
        let segs = plan_segments(&a, 16000, 30.0);
        assert!(segs.len() >= 3);
        for (s, e) in &segs {
            assert!(e - s <= 30 * 16000 + 320);
        }
        assert_eq!(segs.last().unwrap().1, a.len());
    }
}
