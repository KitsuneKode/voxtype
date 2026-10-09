//! Live transcript preview file for streaming sessions
//!
//! Mirrors the in-progress transcript of a streaming session into a file
//! (`[output] streaming_preview_file`) so a status bar or OSD can show the
//! words as they are recognized, without anything reaching the cursor.
//!
//! Each write replaces the file atomically (stage a sibling, then rename),
//! so a reader never sees a half-written transcript. Writes are throttled to
//! one per [`MIN_WRITE_INTERVAL`]: an update that arrives sooner is held as
//! pending and written by [`PreviewFile::flush`] once
//! [`PreviewFile::flush_deadline`] passes, so the latest text always lands.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// At most ~20 writes per second.
pub const MIN_WRITE_INTERVAL: Duration = Duration::from_millis(50);

/// One streaming session's preview file.
pub struct PreviewFile {
    path: PathBuf,
    /// What the file currently holds, to skip no-op rewrites.
    written: String,
    last_write: Option<Instant>,
    /// Newest text not yet written because of the throttle.
    pending: Option<String>,
}

impl PreviewFile {
    /// Start a session's preview at `path`, emptying any leftover content.
    pub fn new(path: PathBuf) -> Self {
        let mut preview = Self {
            path,
            written: String::new(),
            last_write: None,
            pending: None,
        };
        preview.write_now(String::new(), None);
        preview
    }

    /// Publish `text` as the current transcript, now if the throttle allows,
    /// otherwise at the next [`flush`](Self::flush).
    pub fn update(&mut self, text: &str) {
        self.update_at(text, Instant::now());
    }

    fn update_at(&mut self, text: &str, now: Instant) {
        if self.pending.is_none() && text == self.written {
            return;
        }
        match self.last_write {
            Some(at) if now.duration_since(at) < MIN_WRITE_INTERVAL => {
                self.pending = Some(text.to_string());
            }
            _ => self.write_now(text.to_string(), Some(now)),
        }
    }

    /// When the pending update becomes writable, if one is waiting.
    pub fn flush_deadline(&self) -> Option<Instant> {
        self.pending.as_ref()?;
        Some(
            self.last_write
                .map_or_else(Instant::now, |at| at + MIN_WRITE_INTERVAL),
        )
    }

    /// Write the pending update, if any.
    pub fn flush(&mut self) {
        if let Some(text) = self.pending.take() {
            self.write_now(text, Some(Instant::now()));
        }
    }

    /// Empty the file at session end, dropping any pending update.
    pub fn clear(&mut self) {
        self.pending = None;
        self.write_now(String::new(), Some(Instant::now()));
    }

    fn write_now(&mut self, text: String, at: Option<Instant>) {
        self.pending = None;
        if let Err(e) = write_atomic(&self.path, &text) {
            tracing::warn!("Failed to write streaming preview {:?}: {}", self.path, e);
        }
        self.written = text;
        self.last_write = at;
    }
}

fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut staged = path.as_os_str().to_os_string();
    staged.push(".tmp");
    let staged = PathBuf::from(staged);
    std::fs::write(&staged, text)?;
    std::fs::rename(&staged, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&staged);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "voxtype-preview-test-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("live_text")
    }

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn new_empties_leftover_content() {
        let path = temp_path("new");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "stale words from last time").unwrap();
        let _preview = PreviewFile::new(path.clone());
        assert_eq!(read(&path), "");
    }

    #[test]
    fn first_update_writes_immediately() {
        let path = temp_path("first");
        let mut preview = PreviewFile::new(path.clone());
        preview.update("hello");
        assert_eq!(read(&path), "hello");
        assert!(preview.flush_deadline().is_none());
    }

    #[test]
    fn rapid_updates_are_held_then_flushed_with_the_latest_text() {
        let path = temp_path("throttle");
        let mut preview = PreviewFile::new(path.clone());
        let t0 = Instant::now();
        preview.update_at("hello", t0);
        preview.update_at("hello wor", t0 + Duration::from_millis(10));
        preview.update_at("hello world", t0 + Duration::from_millis(20));
        assert_eq!(read(&path), "hello", "throttled updates must not write");
        assert_eq!(preview.flush_deadline(), Some(t0 + MIN_WRITE_INTERVAL));

        preview.flush();
        assert_eq!(read(&path), "hello world");
        assert!(preview.flush_deadline().is_none());
    }

    #[test]
    fn update_after_the_interval_writes_immediately() {
        let path = temp_path("interval");
        let mut preview = PreviewFile::new(path.clone());
        let t0 = Instant::now();
        preview.update_at("one", t0);
        preview.update_at("one two", t0 + MIN_WRITE_INTERVAL);
        assert_eq!(read(&path), "one two");
    }

    #[test]
    fn clear_empties_the_file_and_drops_pending() {
        let path = temp_path("clear");
        let mut preview = PreviewFile::new(path.clone());
        let t0 = Instant::now();
        preview.update_at("one", t0);
        preview.update_at("one two", t0 + Duration::from_millis(1));
        preview.clear();
        assert_eq!(read(&path), "");
        assert!(preview.flush_deadline().is_none());
        assert!(!path.with_file_name("live_text.tmp").exists());
    }
}
