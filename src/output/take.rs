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
        if id.is_empty() || !id.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let dir = runtime.join("takes").join(id);
        // Shuush keeps the shared ID for its own async helpers. Claim it once
        // so a later ordinary hotkey recording doesn't reuse the old target.
        if fs::create_dir_all(&dir).is_err() {
            return None;
        }
        if fs::File::create_new(dir.join("claimed"))
            .is_err_and(|e| e.kind() == std::io::ErrorKind::AlreadyExists)
        {
            return None;
        }
        let window: Option<serde_json::Value> = fs::read(dir.join("target_window"))
            .ok()
            .and_then(|s| serde_json::from_slice(&s).ok());
        let target = window.and_then(|w| {
            let address = w["address"].as_str()?.trim();
            if address.is_empty() || address == "0x0" {
                return None;
            }
            Some((
                address.to_owned(),
                w["pid"].as_u64().filter(|pid| *pid != 0),
            ))
        });
        let private_at_start =
            dir.join("pp_private").exists() || runtime.join("sensitive_field").exists();
        if private_at_start {
            let _ = fs::write(dir.join("pp_private"), b"1");
        }
        Some(Self {
            id: id.to_owned(),
            dir,
            target,
            private_at_start,
        })
    }
    pub fn private(&self) -> bool {
        self.private_at_start || self.dir.join("pp_private").exists()
    }
    pub fn disposition(&self) -> Option<OutcomeKind> {
        match fs::read_to_string(self.dir.join("disposition"))
            .ok()?
            .trim()
        {
            "private_not_inserted" => Some(OutcomeKind::PrivateNotInserted),
            "recovered" if self.private() => Some(OutcomeKind::PrivateNotInserted),
            "recovered" => Some(OutcomeKind::Recovered),
            _ => None,
        }
    }
    pub fn target_lost(&self, text: &str) -> OutcomeKind {
        if let Some(outcome) = self.disposition() {
            return outcome;
        }
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
            Err(e) => {
                tracing::warn!("Cannot save take for recovery: {e}");
                OutcomeKind::Failed
            }
        }
    }
    async fn target_focused(&self) -> bool {
        let Some((expected, pid)) = &self.target else {
            return false;
        };
        super::active_window::focused_identity()
            .await
            .is_some_and(|(address, current_pid)| {
                address == *expected && pid.is_none_or(|p| Some(p) == current_pid)
            })
    }
}

// Individual drivers can check again after their own delay, immediately
// before invoking the compositor or keyboard tool.
tokio::task_local! { pub(crate) static DELIVERY_TAKE: Option<TakeIdentity>; }

pub(crate) async fn check_target() -> Result<(), OutputError> {
    if let Some(take) = DELIVERY_TAKE.try_with(Clone::clone).ok().flatten() {
        if take.disposition().is_some() || !take.target_focused().await {
            return Err(OutputError::TargetChanged);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn start(runtime: &Path, id: &str, private: bool) {
        let dir = runtime.join("takes").join(id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(runtime.join("take_id"), id).unwrap();
        fs::write(
            dir.join("target_window"),
            format!(r#"{{"address":"0x{id}","pid":{id}}}"#),
        )
        .unwrap();
        if private {
            fs::write(dir.join("pp_private"), "").unwrap();
        }
    }

    #[test]
    fn a_take_keeps_its_id_target_and_privacy_when_the_next_one_starts() {
        let runtime = tempfile::tempdir().unwrap();
        start(runtime.path(), "1", false);
        let first = TakeIdentity::capture(runtime.path()).unwrap();

        // The second take starts while the first is post-processing, in a
        // password field this time.
        start(runtime.path(), "2", true);
        fs::write(runtime.path().join("sensitive_field"), "1").unwrap();
        let second = TakeIdentity::capture(runtime.path()).unwrap();

        assert_eq!(first.id, "1");
        assert_eq!(first.target, Some(("0x1".into(), Some(1))));
        assert!(
            !first.private(),
            "the next take's privacy leaked into this one"
        );
        assert_eq!(second.id, "2");
        assert_eq!(second.target, Some(("0x2".into(), Some(2))));
        assert!(second.private());
    }

    #[test]
    fn a_sensitive_field_at_the_start_makes_the_take_private() {
        let runtime = tempfile::tempdir().unwrap();
        start(runtime.path(), "5", false);
        fs::write(runtime.path().join("sensitive_field"), "1").unwrap();
        let take = TakeIdentity::capture(runtime.path()).unwrap();
        fs::remove_file(runtime.path().join("sensitive_field")).unwrap();
        assert!(take.private(), "the take forgot the field was sensitive");
        assert!(take.dir.join("pp_private").exists());
    }

    #[test]
    fn an_id_is_claimed_once_and_a_malformed_one_is_ignored() {
        let runtime = tempfile::tempdir().unwrap();
        start(runtime.path(), "3", false);
        assert!(TakeIdentity::capture(runtime.path()).is_some());
        assert!(
            TakeIdentity::capture(runtime.path()).is_none(),
            "a later recording reused the last take's id"
        );
        for bad in ["", "../x", "12a", " "] {
            fs::write(runtime.path().join("take_id"), bad).unwrap();
            assert!(TakeIdentity::capture(runtime.path()).is_none(), "{bad:?}");
        }
        fs::remove_file(runtime.path().join("take_id")).unwrap();
        assert!(TakeIdentity::capture(runtime.path()).is_none());
    }

    #[test]
    fn a_lost_target_keeps_ordinary_text_for_recovery_only() {
        let runtime = tempfile::tempdir().unwrap();
        start(runtime.path(), "7", false);
        let take = TakeIdentity::capture(runtime.path()).unwrap();
        assert_eq!(take.target_lost("hello there"), OutcomeKind::Recovered);
        let saved = take.dir.join("recovered.txt");
        assert_eq!(fs::read_to_string(&saved).unwrap(), "hello there");
        assert_eq!(
            fs::metadata(&saved).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::read_to_string(take.dir.join("disposition")).unwrap(),
            "recovered"
        );
        assert_eq!(take.disposition(), Some(OutcomeKind::Recovered));

        start(runtime.path(), "8", true);
        let private = TakeIdentity::capture(runtime.path()).unwrap();
        assert_eq!(
            private.target_lost("hunter two"),
            OutcomeKind::PrivateNotInserted
        );
        assert!(!private.dir.join("recovered.txt").exists());
        assert_eq!(
            fs::read_to_string(private.dir.join("disposition")).unwrap(),
            "private_not_inserted"
        );
    }

    #[test]
    fn shuush_recovering_a_private_take_still_reads_as_private() {
        let runtime = tempfile::tempdir().unwrap();
        start(runtime.path(), "9", true);
        let take = TakeIdentity::capture(runtime.path()).unwrap();
        assert_eq!(take.disposition(), None);
        fs::write(take.dir.join("disposition"), "recovered\n").unwrap();
        assert_eq!(take.disposition(), Some(OutcomeKind::PrivateNotInserted));
    }

    #[tokio::test]
    async fn delivery_outside_an_identified_take_is_never_blocked() {
        assert!(check_target().await.is_ok());
        assert!(DELIVERY_TAKE.scope(None, check_target()).await.is_ok());
    }

    #[tokio::test]
    async fn a_take_without_a_known_window_or_with_a_disposition_is_not_delivered() {
        let runtime = tempfile::tempdir().unwrap();
        start(runtime.path(), "4", false);
        fs::write(runtime.path().join("takes/4/target_window"), "{}").unwrap();
        let take = TakeIdentity::capture(runtime.path()).unwrap();
        assert!(matches!(
            DELIVERY_TAKE.scope(Some(take), check_target()).await,
            Err(OutputError::TargetChanged)
        ));
    }
}
