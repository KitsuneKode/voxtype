//! The last cancelled take, kept for a few minutes so a cancel pressed by
//! mistake can be undone with `voxtype record recover`.
//!
//! Only a take long enough to hold something is kept: at least
//! [`MIN_SECS`] long with [`MIN_VOICED_SECS`] of it above the speech level.
//! It is written to `$XDG_RUNTIME_DIR/voxtype/cancelled_take.wav` (memory, not
//! disk, on a systemd login), readable only by the user, replaced by the next
//! cancel, removed when it is recovered and treated as gone after
//! [`KEEP_SECS`]. Without `XDG_RUNTIME_DIR` nothing is kept: the voice never
//! goes to `/tmp`.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// A take shorter than this is a slip of the key, not something to bring back.
pub const MIN_SECS: f32 = 2.0;
/// How much of it has to sound like speech.
pub const MIN_VOICED_SECS: f32 = 1.0;
/// How long a cancelled take can be brought back.
pub const KEEP_SECS: u64 = 600;

const RATE: u32 = 16000;
const FRAME: usize = 320;
const VOICED_RMS: f32 = 0.012;

/// Where the take is kept, or `None` without a per-user runtime directory.
pub fn path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_RUNTIME_DIR").filter(|v| !v.is_empty())?;
    Some(PathBuf::from(base).join("voxtype").join("cancelled_take.wav"))
}

/// Seconds of `samples` (16 kHz mono) loud enough to be speech, by 20 ms frames.
pub fn voiced_secs(samples: &[f32]) -> f32 {
    let voiced = samples
        .chunks(FRAME)
        .filter(|f| {
            let energy: f32 = f.iter().map(|s| s * s).sum::<f32>() / f.len().max(1) as f32;
            energy.sqrt() >= VOICED_RMS
        })
        .count();
    voiced as f32 * FRAME as f32 / RATE as f32
}

/// Whether a cancelled take is worth keeping.
pub fn worth_keeping(samples: &[f32]) -> bool {
    samples.len() as f32 / RATE as f32 >= MIN_SECS && voiced_secs(samples) >= MIN_VOICED_SECS
}

/// Keep `samples` if they are worth it. Returns whether they were kept.
pub fn keep(samples: &[f32]) -> bool {
    if !worth_keeping(samples) {
        return false;
    }
    let Some(dest) = path() else { return false };
    match write(&dest, samples) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!("could not keep the cancelled take: {e}");
            let _ = std::fs::remove_file(&dest);
            false
        }
    }
}

fn write(dest: &std::path::Path, samples: &[f32]) -> anyhow::Result<()> {
    let dir = dest.parent().ok_or_else(|| anyhow::anyhow!("no parent"))?;
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join("cancelled_take.wav.tmp");
    let _ = std::fs::remove_file(&tmp);
    {
        let file = open_private(&tmp)?;
        let spec = hound::WavSpec { channels: 1, sample_rate: RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::new(std::io::BufWriter::new(file), spec)?;
        for &s in samples {
            writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        writer.finalize()?;
    }
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[cfg(unix)]
fn open_private(p: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(p)
}

#[cfg(not(unix))]
fn open_private(p: &std::path::Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new().write(true).create_new(true).open(p)
}

/// Whether a take is there to bring back (and still within [`KEEP_SECS`]).
pub fn available() -> bool {
    path().is_some_and(|p| fresh(&p))
}

fn fresh(p: &std::path::Path) -> bool {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| SystemTime::now().duration_since(t).ok())
        .is_some_and(|age| age <= Duration::from_secs(KEEP_SECS))
}

/// Take the kept samples out (the file is removed either way).
pub fn take() -> Option<Vec<f32>> {
    let p = path()?;
    let ok = fresh(&p);
    let samples = ok.then(|| read(&p)).flatten();
    let _ = std::fs::remove_file(&p);
    samples.filter(|s| !s.is_empty())
}

fn read(p: &std::path::Path) -> Option<Vec<f32>> {
    let mut reader = hound::WavReader::open(p).ok()?;
    if reader.spec().sample_rate != RATE || reader.spec().channels != 1 {
        return None;
    }
    reader.samples::<i16>().map(|s| s.ok().map(|v| v as f32 / 32767.0)).collect()
}

/// Remove a kept take, if any.
/// At start: a take older than its window (the daemon ended before its timer) goes.
pub fn sweep() {
    if let Some(p) = path() {
        if p.exists() && !fresh(&p) {
            let _ = std::fs::remove_file(&p);
        }
    }
}

pub fn forget() {
    if let Some(p) = path() {
        let _ = std::fs::remove_file(p);
    }
}

/// Remove the take once it is too old, unless a newer one replaced it.
pub fn expire_later() {
    let Some(p) = path() else { return };
    let stamp = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(KEEP_SECS)).await;
        if std::fs::metadata(&p).and_then(|m| m.modified()).ok() == stamp {
            let _ = std::fs::remove_file(&p);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        (0..(secs * RATE as f32) as usize).map(|i| amp * (i as f32 * 0.05).sin()).collect()
    }

    #[test]
    fn an_old_take_is_swept_at_start() {
        let dir = std::env::temp_dir().join(format!("vt-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("cancelled.wav");
        std::fs::write(&p, b"x").unwrap();
        let old = std::time::SystemTime::now() - Duration::from_secs(KEEP_SECS + 60);
        std::fs::File::options().write(true).open(&p).unwrap().set_modified(old).unwrap();
        assert!(!fresh(&p));
        std::fs::write(dir.join("new.wav"), b"x").unwrap();
        assert!(fresh(&dir.join("new.wav")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_slip_of_the_key_is_not_kept() {
        assert!(!worth_keeping(&tone(1.2, 0.3)), "too short");
        assert!(!worth_keeping(&tone(6.0, 0.001)), "long but silent");
        assert!(worth_keeping(&tone(3.0, 0.3)));
    }

    #[test]
    fn it_is_kept_privately_and_given_back_once() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("XDG_RUNTIME_DIR", dir.path());
        let samples = tone(3.0, 0.3);
        assert!(keep(&samples));
        let p = path().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert!(available());
        let back = take().unwrap();
        assert_eq!(back.len(), samples.len());
        assert!(!p.exists() && take().is_none(), "recovered once, then gone");
        assert!(keep(&samples));
        assert!(!keep(&tone(0.5, 0.3)) && available(), "a slip of the key after a real cancel leaves that take there");
    }
}
