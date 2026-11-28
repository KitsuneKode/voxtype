//! Delivery contract for a Shuush recording. Only the start reads the shared ID.
use crate::error::OutputError;
use crate::osd::session::OutcomeKind;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub(crate) struct TakeIdentity {
    pub id: String,
    pub dir: PathBuf,
    target: Option<(String, Option<u64>)>,
    private_at_start: bool,
}

impl TakeIdentity {
    pub fn capture(runtime: &Path) -> Option<Self> {
        let id = fs::read_to_string(runtime.join("take_id")).ok()?;
        let id = id.trim();
        if id.is_empty() || !id.bytes().all(|c| c.is_ascii_digit()) { return None; }
        let dir = runtime.join("takes").join(id);
        // Shuush keeps the shared ID for its own async helpers. Claim it once
        // so a later ordinary hotkey recording doesn't reuse the old target.
        if fs::create_dir_all(&dir).is_err() { return None; }
        if fs::File::create_new(dir.join("claimed")).is_err_and(|e| e.kind() == std::io::ErrorKind::AlreadyExists) { return None; }
        let window: Option<serde_json::Value> = fs::read(dir.join("target_window")).ok().and_then(|s| serde_json::from_slice(&s).ok());
        let target = window.and_then(|w| {
            let address = w["address"].as_str()?.trim();
            if address.is_empty() || address == "0x0" { return None; }
            Some((address.to_owned(), w["pid"].as_u64().filter(|pid| *pid != 0)))
        });
        let private_at_start = dir.join("pp_private").exists() || runtime.join("sensitive_field").exists();
        if private_at_start { let _ = fs::write(dir.join("pp_private"), b"1"); }
        Some(Self {id: id.to_owned(), dir, target, private_at_start})
    }
    pub fn private(&self) -> bool {
        self.private_at_start || self.dir.join("pp_private").exists()
    }
    pub fn disposition(&self) -> Option<OutcomeKind> {
        match fs::read_to_string(self.dir.join("disposition")).ok()?.trim() {
            "private_not_inserted" => Some(OutcomeKind::PrivateNotInserted),
            "recovered" if self.private() => Some(OutcomeKind::PrivateNotInserted),
            "recovered" => Some(OutcomeKind::Recovered),
            _ => None
        }
    }
    pub fn target_lost(&self, text: &str) -> OutcomeKind {
        if let Some(outcome) = self.disposition() { return outcome; }
        if self.private() {
            let _ = fs::write(self.dir.join("disposition"), b"private_not_inserted");
            return OutcomeKind::PrivateNotInserted;
        }
        let save = || -> std::io::Result<()> {
            let mut file = tempfile::NamedTempFile::new_in(&self.dir)?;
            file.write_all(text.as_bytes())?;
            file.persist(self.dir.join("recovered.txt"))?;
            fs::write(self.dir.join("disposition"), b"recovered")
        };
        match save() {
            Ok(()) => OutcomeKind::Recovered,
            Err(e) => { tracing::warn!("Cannot save take for recovery: {e}"); OutcomeKind::Failed }
        }
    }
    async fn target_focused(&self) -> bool {
        let Some((expected, pid)) = &self.target else { return false; };
        super::active_window::focused_identity().await.is_some_and(|(address, current_pid)| {
            address == *expected && pid.is_none_or(|p| Some(p) == current_pid)
        })
    }
}

// Individual drivers can check again after their own delay, immediately
// before invoking the compositor or keyboard tool.
tokio::task_local! { pub(crate) static DELIVERY_TAKE: Option<TakeIdentity>; }

pub(crate) async fn check_target() -> Result<(), OutputError> {
    if let Some(take) = DELIVERY_TAKE.try_with(Clone::clone).ok().flatten() {
        if take.disposition().is_some() || !take.target_focused().await { return Err(OutputError::TargetChanged); }
    }
    Ok(())
}
