//! Post-processing command execution
//!
//! Pipes transcribed text through an external command for cleanup/formatting.
//! Commonly used with local LLMs (Ollama, llama.cpp) or text processing tools.
//!
//! # Example Configuration
//!
//! ```toml
//! [output.post_process]
//! command = "ollama run llama3.2:1b 'Clean up this dictation:'"
//! timeout_ms = 30000
//! ```
//!
//! The command receives the transcribed text on stdin and should output
//! the processed text on stdout. On any failure, the original text is used.
//!
//! With `streaming = true` the command is started with
//! `VOXTYPE_STREAM_OUTPUT=1` and its stdout is read as it arrives
//! ([`PostProcessor::process_streaming`]), so the daemon can type each piece
//! while the command is still running.

use crate::config::PostProcessConfig;
use serde::Deserialize;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::timeout;

/// Post-processor that runs an external command on transcribed text
pub struct PostProcessor {
    command: String,
    timeout: Duration,
    trim: bool,
    fallback_on_empty: bool,
    streaming: bool,
    edit_list: bool,
}

/// How a streamed post-process run ([`PostProcessor::process_streaming`])
/// ended. The pieces themselves went out over the channel as they were read.
#[derive(Debug)]
pub enum StreamEnd {
    /// The command exited 0; what it printed is the result.
    Done,
    /// It failed (timeout, non-zero exit, output that is not UTF-8) after
    /// the pieces already sent, if any.
    Failed(PostProcessError),
}

impl PostProcessor {
    /// Create a new post-processor from configuration
    pub fn new(config: &PostProcessConfig) -> Self {
        Self {
            command: config.command.clone(),
            timeout: Duration::from_millis(config.timeout_ms),
            trim: config.trim,
            fallback_on_empty: config.fallback_on_empty,
            streaming: config.streaming,
            edit_list: config.edit_list,
        }
    }

    /// Whether the command prints its output progressively (see
    /// [`Self::process_streaming`]).
    pub fn streaming(&self) -> bool {
        self.streaming
    }

    /// Whether whitespace around the command's output is dropped.
    pub fn trims(&self) -> bool {
        self.trim
    }

    /// Whether an empty result means "use the original text".
    pub fn falls_back_on_empty(&self) -> bool {
        self.fallback_on_empty
    }

    /// Run the command with `VOXTYPE_STREAM_OUTPUT=1` and send its stdout to
    /// `pieces` as it is read, in order, each piece valid UTF-8 (a character
    /// split across two reads goes out whole with the second). Returns once
    /// the command has exited or failed. Dropping the future kills the
    /// command, like the batch path.
    pub async fn process_streaming(
        &self,
        text: &str,
        context: Option<&str>,
        pieces: mpsc::UnboundedSender<String>,
    ) -> StreamEnd {
        match self.stream_command(text, context, &pieces).await {
            Ok(()) => StreamEnd::Done,
            Err(e) => StreamEnd::Failed(e),
        }
    }

    async fn stream_command(
        &self,
        text: &str,
        context: Option<&str>,
        pieces: &mpsc::UnboundedSender<String>,
    ) -> Result<(), PostProcessError> {
        let deadline = tokio::time::Instant::now() + self.timeout;
        let timed_out = || PostProcessError::Timeout(self.timeout.as_secs());
        let mut cmd = self.command(context);
        cmd.env("VOXTYPE_STREAM_OUTPUT", "1");
        let mut child = cmd
            .spawn()
            .map_err(|e| PostProcessError::SpawnFailed(e.to_string()))?;
        let mut group = ProcessGroupGuard(child.id());

        if let Some(mut stdin) = child.stdin.take() {
            let _ = tokio::time::timeout_at(deadline, stdin.write_all(text.as_bytes())).await;
            drop(stdin);
        }
        // Drain stderr on the side so a chatty command cannot block on a
        // full pipe while we wait for its stdout.
        let stderr_task = child.stderr.take().map(|mut err| {
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let _ = err.read_to_end(&mut buf).await;
                String::from_utf8_lossy(&buf).trim().to_string()
            })
        });

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| PostProcessError::WaitFailed("no stdout".into()))?;
        let mut buf = [0u8; 4096];
        let mut carry: Vec<u8> = Vec::new();
        loop {
            let n = tokio::time::timeout_at(deadline, stdout.read(&mut buf))
                .await
                .map_err(|_| timed_out())?
                .map_err(|e| PostProcessError::WaitFailed(e.to_string()))?;
            if n == 0 {
                break;
            }
            carry.extend_from_slice(&buf[..n]);
            let valid = match std::str::from_utf8(&carry) {
                Ok(s) => s.len(),
                // Incomplete character at the end: keep it for the next read.
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(e) => return Err(PostProcessError::InvalidUtf8(e.to_string())),
            };
            if valid > 0 {
                let rest = carry.split_off(valid);
                let piece = String::from_utf8(std::mem::replace(&mut carry, rest))
                    .expect("validated above");
                let _ = pieces.send(piece);
            }
        }
        if !carry.is_empty() {
            return Err(PostProcessError::InvalidUtf8(
                "output ended inside a character".into(),
            ));
        }

        let status = tokio::time::timeout_at(deadline, child.wait())
            .await
            .map_err(|_| timed_out())?
            .map_err(|e| PostProcessError::WaitFailed(e.to_string()))?;
        group.disarm();
        if !status.success() {
            let stderr = match stderr_task {
                Some(t) => t.await.unwrap_or_default(),
                None => String::new(),
            };
            return Err(PostProcessError::NonZeroExit {
                code: status.code(),
                stderr,
            });
        }
        Ok(())
    }

    fn command(&self, context: Option<&str>) -> Command {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", &self.command])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Own process group, so the guard below can kill what `sh -c`
            // started (a curl to an LLM API, say) and not just the shell.
            .process_group(0);

        // Always clear to prevent inheriting stale context from parent environment
        cmd.env_remove("VOXTYPE_CONTEXT");
        cmd.env_remove("VOXTYPE_STREAM_OUTPUT");
        if let Some(ctx) = context {
            cmd.env("VOXTYPE_CONTEXT", ctx);
        }
        cmd
    }

    /// Process text with optional context from a previous chunk
    ///
    /// When context is provided, it is passed via the VOXTYPE_CONTEXT environment
    /// variable so the post-processing command can use it for continuity.
    /// Stdin always contains only the current text, keeping existing scripts compatible.
    /// Returns the processed text on success, or the original text on any failure.
    pub async fn process_with_context(&self, text: &str, context: Option<&str>) -> String {
        match self.execute_command_with_env(text, context).await {
            Ok(command_output) => {
                let processed = if self.edit_list {
                    match apply_edit_list(text, &command_output) {
                        Ok(processed) => processed,
                        Err(e) => {
                            tracing::warn!(
                                "Post-process command returned an invalid edit list: {}, using original text",
                                e
                            );
                            return text.to_string();
                        }
                    }
                } else {
                    command_output
                };
                if processed.is_empty() && self.fallback_on_empty {
                    tracing::warn!(
                        "Post-process command returned empty output, using original text"
                    );
                    text.to_string()
                } else if processed.is_empty() {
                    tracing::debug!("Post-process command returned empty output");
                    String::new()
                } else {
                    tracing::debug!(
                        "Post-processed ({} -> {} chars)",
                        text.len(),
                        processed.len()
                    );
                    processed
                }
            }
            Err(e) => {
                tracing::warn!("Post-process command failed: {}, using original text", e);
                text.to_string()
            }
        }
    }

    /// Process text through the external command
    ///
    /// Returns the processed text on success, or the original text on any failure.
    /// This ensures voice-to-text always produces output even when post-processing fails.
    pub async fn process(&self, text: &str) -> String {
        self.process_with_context(text, None).await
    }

    async fn execute_command_with_env(
        &self,
        text: &str,
        context: Option<&str>,
    ) -> Result<String, PostProcessError> {
        let mut cmd = self.command(context);
        let mut child = cmd
            .spawn()
            .map_err(|e| PostProcessError::SpawnFailed(e.to_string()))?;
        let mut group = ProcessGroupGuard(child.id());

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            // Ignore write errors: the command may not read stdin or may exit
            // before we finish writing (e.g., `echo` or `head -1`). The command's
            // exit code and stdout output determine success, not whether it
            // consumed all of stdin.
            let _ = stdin.write_all(text.as_bytes()).await;
            drop(stdin);
        }

        // Wait for completion with timeout
        let output = timeout(self.timeout, child.wait_with_output())
            .await
            .map_err(|_| PostProcessError::Timeout(self.timeout.as_secs()))?
            .map_err(|e| PostProcessError::WaitFailed(e.to_string()))?;
        group.disarm();

        // Check exit status
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PostProcessError::NonZeroExit {
                code: output.status.code(),
                stderr: stderr.trim().to_string(),
            });
        }

        // Parse stdout as UTF-8
        let processed = String::from_utf8(output.stdout)
            .map_err(|e| PostProcessError::InvalidUtf8(e.to_string()))?;

        if self.trim {
            Ok(processed.trim().to_string())
        } else {
            // Only strip trailing newlines (artifact of shell output), preserve other whitespace
            Ok(processed.trim_end_matches('\n').to_string())
        }
    }
}

/// Kills a command's process group when dropped while still armed.
///
/// Covers both ways the command can be abandoned: the timeout firing, and
/// the daemon dropping the whole future because the user cancelled or asked
/// for the raw transcript. Without it the command ran on unobserved.
struct ProcessGroupGuard(Option<u32>);

impl ProcessGroupGuard {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            // The shell is not reaped until wait completes, so its pid (the
            // group id) cannot have been reused while the guard is armed.
            unsafe {
                libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextEdit {
    find: String,
    replace: String,
    reason: String,
}

/// Apply a command's JSON edit list without accepting any undeclared output.
///
/// Every target must occur exactly once in the original text. Resolving all
/// spans before applying any edit prevents one replacement from creating the
/// target for a later edit, and rejecting overlaps keeps the result
/// deterministic regardless of list order.
fn apply_edit_list(text: &str, output: &str) -> Result<String, PostProcessError> {
    let edits: Vec<TextEdit> = serde_json::from_str(output)
        .map_err(|e| PostProcessError::InvalidEditList(e.to_string()))?;
    let mut resolved = Vec::with_capacity(edits.len());

    for edit in edits {
        if edit.find.is_empty() {
            return Err(PostProcessError::InvalidEditList(
                "find must not be empty".to_string(),
            ));
        }
        if edit.reason.trim().is_empty() {
            return Err(PostProcessError::InvalidEditList(
                "reason must not be empty".to_string(),
            ));
        }

        // Count scalar values, not UTF-8 bytes: non-ASCII edits get the same
        // budget as English ones. Cap spans before the quadratic distance check.
        let old_len = edit.find.chars().count();
        let new_len = edit.replace.chars().count();
        if old_len > 4096
            || new_len > old_len.saturating_mul(2).saturating_add(8)
            || strsim::levenshtein(&edit.find, &edit.replace) > (old_len / 2).max(4)
        {
            return Err(PostProcessError::InvalidEditList(
                "replacement exceeds the cleanup edit budget".to_string(),
            ));
        }

        let Some(start) = text.find(&edit.find) else {
            return Err(PostProcessError::InvalidEditList(format!(
                "find target {:?} is not in the transcription",
                edit.find
            )));
        };
        // Search from the next character, not the end of the first match:
        // `aa` is ambiguous in `aaa` even though its occurrences overlap.
        let next = start + text[start..].chars().next().unwrap().len_utf8();
        if text[next..].contains(&edit.find) {
            return Err(PostProcessError::InvalidEditList(format!(
                "find target {:?} is ambiguous",
                edit.find
            )));
        }
        resolved.push((start, start + edit.find.len(), edit.replace));
    }

    resolved.sort_by_key(|(start, _, _)| *start);
    for pair in resolved.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(PostProcessError::InvalidEditList(
                "edits overlap".to_string(),
            ));
        }
    }

    let mut result = text.to_string();
    for (start, end, replacement) in resolved.into_iter().rev() {
        result.replace_range(start..end, &replacement);
    }
    let original_len = text.chars().count();
    if result.chars().count() > original_len.saturating_add((original_len / 5).max(8)) {
        return Err(PostProcessError::InvalidEditList(
            "edit list adds too much content".to_string(),
        ));
    }
    Ok(result)
}

/// Errors that can occur during post-processing
#[derive(Debug)]
pub enum PostProcessError {
    /// Failed to spawn the command process
    SpawnFailed(String),
    /// Failed to write text to stdin
    WriteFailed(String),
    /// Command timed out
    Timeout(u64),
    /// Failed to wait for command completion
    WaitFailed(String),
    /// Command exited with non-zero status
    NonZeroExit { code: Option<i32>, stderr: String },
    /// Command output was not valid UTF-8
    InvalidUtf8(String),
    /// Command output did not satisfy the edit-list contract
    InvalidEditList(String),
}

impl std::fmt::Display for PostProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpawnFailed(e) => write!(f, "failed to spawn command: {}", e),
            Self::WriteFailed(e) => write!(f, "failed to write to stdin: {}", e),
            Self::Timeout(secs) => write!(f, "command timed out after {}s", secs),
            Self::WaitFailed(e) => write!(f, "failed to wait for command: {}", e),
            Self::NonZeroExit { code, stderr } => {
                if stderr.is_empty() {
                    write!(f, "command exited with code {:?}", code)
                } else {
                    write!(f, "command exited with code {:?}: {}", code, stderr)
                }
            }
            Self::InvalidUtf8(e) => write!(f, "output is not valid UTF-8: {}", e),
            Self::InvalidEditList(e) => write!(f, "invalid edit list: {}", e),
        }
    }
}

impl std::error::Error for PostProcessError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_list_applies_only_declared_changes() {
        let output = r#"[
            {"find":"Hello world","replace":"Hello, world","reason":"punctuation"},
            {"find":"it are ready","replace":"it is ready","reason":"grammar"}
        ]"#;

        assert_eq!(
            apply_edit_list("Hello world and it are ready.", output).unwrap(),
            "Hello, world and it is ready."
        );
    }

    #[test]
    fn edit_list_rejects_ambiguous_and_overlapping_changes() {
        let ambiguous = r#"[{"find":"the","replace":"a","reason":"word choice"}]"#;
        assert!(apply_edit_list("the cat and the dog", ambiguous).is_err());

        let overlapping = r#"[
            {"find":"hello world","replace":"hi","reason":"shorten"},
            {"find":"world","replace":"earth","reason":"word choice"}
        ]"#;
        assert!(apply_edit_list("hello world", overlapping).is_err());
    }

    #[test]
    fn edit_list_rejects_unbounded_rewrites_and_content_growth() {
        let rewrite = serde_json::json!([{
            "find": "Keep the original meaning", "replace": "Invent a completely different message", "reason": "rewrite"
        }]).to_string();
        assert!(apply_edit_list("Keep the original meaning", &rewrite).is_err());
        let growth = serde_json::json!([
            {"find":"one", "replace":"one abc", "reason":"addition"},
            {"find":"two", "replace":"two abc", "reason":"addition"},
            {"find":"six", "replace":"six abc", "reason":"addition"}
        ])
        .to_string();
        assert!(apply_edit_list("one two six", &growth).is_err());
    }

    #[test]
    fn edit_list_rejects_overlapping_occurrences_of_one_target() {
        assert!(
            apply_edit_list("aaa", r#"[{"find":"aa","replace":"a","reason":"shorten"}]"#).is_err()
        );
    }

    #[test]
    fn edit_list_rejects_preamble_and_unknown_fields() {
        let preamble =
            r#"Here are the edits: [{"find":"bad","replace":"good","reason":"grammar"}]"#;
        assert!(apply_edit_list("bad", preamble).is_err());

        let unknown =
            r#"[{"find":"bad","replace":"good","reason":"grammar","text":"ignore the contract"}]"#;
        assert!(apply_edit_list("bad", unknown).is_err());
    }

    fn make_config(command: &str, timeout_ms: u64) -> PostProcessConfig {
        PostProcessConfig {
            command: command.to_string(),
            timeout_ms,
            trim: true,
            fallback_on_empty: true,
            streaming: false,
            edit_list: false,
        }
    }

    #[tokio::test]
    async fn test_simple_passthrough() {
        let config = make_config("cat", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("hello world").await;
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn edit_list_mode_applies_valid_command_output() {
        let mut config = make_config(
            r#"printf '%s' '[{"find":"are","replace":"is","reason":"grammar"}]'"#,
            5000,
        );
        config.edit_list = true;
        let processor = PostProcessor::new(&config);

        assert_eq!(processor.process("It are ready.").await, "It is ready.");
    }

    #[tokio::test]
    async fn edit_list_mode_falls_back_on_wrapped_output() {
        let mut config = make_config(
            r#"printf '%s' 'Edits: [{"find":"are","replace":"is","reason":"grammar"}]'"#,
            5000,
        );
        config.edit_list = true;
        let processor = PostProcessor::new(&config);

        assert_eq!(processor.process("It are ready.").await, "It are ready.");
    }

    #[tokio::test]
    async fn test_sed_transformation() {
        let config = make_config("sed 's/foo/bar/g'", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("foo bar foo").await;
        assert_eq!(result, "bar bar bar");
    }

    #[tokio::test]
    async fn test_tr_uppercase() {
        let config = make_config("tr '[:lower:]' '[:upper:]'", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("hello world").await;
        assert_eq!(result, "HELLO WORLD");
    }

    #[tokio::test]
    async fn test_timeout_fallback() {
        let config = make_config("sleep 10", 100); // 100ms timeout
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "original text"); // Falls back to original
    }

    /// Pid a test command recorded, waiting for the command to get there.
    #[cfg(target_os = "linux")]
    async fn wait_for_pid(file: &std::path::Path) -> i32 {
        for _ in 0..100 {
            if let Some(pid) = std::fs::read_to_string(file)
                .ok()
                .and_then(|s| s.trim().parse().ok())
            {
                return pid;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("command never wrote its pid to {:?}", file);
    }

    /// A zombie still answers kill(pid, 0), so read the state from /proc.
    #[cfg(target_os = "linux")]
    fn is_running(pid: i32) -> bool {
        std::fs::read_to_string(format!("/proc/{}/stat", pid))
            .ok()
            .and_then(|stat| {
                let state = stat.rsplit(')').next()?.split_whitespace().next()?;
                Some(state != "Z" && state != "X")
            })
            .unwrap_or(false)
    }

    #[cfg(target_os = "linux")]
    async fn assert_stops(pid: i32) {
        for _ in 0..100 {
            if !is_running(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        unsafe { libc::kill(pid, libc::SIGKILL) };
        panic!("process {} the command started outlived it", pid);
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn test_timeout_kills_what_the_command_started() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let command = format!("sleep 30 & echo $! > '{}'; wait", pid_file.display());
        let processor = PostProcessor::new(&make_config(&command, 300));
        assert_eq!(processor.process("original text").await, "original text");
        assert_stops(wait_for_pid(&pid_file).await).await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn test_abandoning_a_run_kills_what_the_command_started() {
        // The daemon drops the run on `record cancel` and on
        // `record stop --no-post-process`; an LLM call the script started
        // must not keep going.
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let command = format!("sleep 30 & echo $! > '{}'; wait", pid_file.display());
        let processor = PostProcessor::new(&make_config(&command, 30_000));
        let pid = {
            let run = processor.process("original text");
            tokio::pin!(run);
            tokio::select! {
                _ = &mut run => panic!("command should still be running"),
                pid = wait_for_pid(&pid_file) => pid,
            }
        };
        assert_stops(pid).await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn test_completed_run_leaves_background_processes_alone() {
        // Scripts fork long-lived helpers on success, e.g. a wl-copy
        // clipboard owner; finishing normally must not take them down.
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let command = format!(
            "sleep 30 </dev/null >/dev/null 2>&1 & echo $! > '{}'; echo done",
            pid_file.display()
        );
        let processor = PostProcessor::new(&make_config(&command, 5000));
        assert_eq!(processor.process("original text").await, "done");
        let pid = wait_for_pid(&pid_file).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        let survived = is_running(pid);
        unsafe { libc::kill(pid, libc::SIGKILL) };
        assert!(
            survived,
            "background process was killed after a successful run"
        );
    }

    #[tokio::test]
    async fn test_command_failure_fallback() {
        let config = make_config("exit 1", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "original text"); // Falls back to original
    }

    #[tokio::test]
    async fn test_empty_output_fallback() {
        // printf '' outputs nothing, which should trigger fallback
        // (echo -n is not portable across platforms)
        let config = make_config("printf ''", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "original text"); // Falls back to original
    }

    #[tokio::test]
    async fn test_command_not_found_fallback() {
        let config = make_config("nonexistent_command_xyz_12345", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "original text"); // Falls back to original
    }

    #[tokio::test]
    async fn test_multiline_input() {
        let config = make_config("cat", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("line one\nline two\nline three").await;
        assert_eq!(result, "line one\nline two\nline three");
    }

    #[tokio::test]
    async fn test_unicode_handling() {
        let config = make_config("cat", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("Hello 世界! 🎉").await;
        assert_eq!(result, "Hello 世界! 🎉");
    }

    #[tokio::test]
    async fn test_whitespace_trimming() {
        // Output has trailing newline which should be trimmed
        // Use printf with \n to be portable across platforms
        let config = make_config("printf 'hello\\n'", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("ignored").await;
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn test_complex_shell_command() {
        // Test that complex shell commands work (pipes, quotes, etc.)
        let config = make_config("echo 'prefix:' && cat", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("test input").await;
        assert_eq!(result, "prefix:\ntest input");
    }

    #[tokio::test]
    async fn test_no_trim_preserves_trailing_space() {
        // When trim = false, trailing spaces from the command should be preserved
        let config = PostProcessConfig {
            command: "printf '%s ' \"$( cat )\"".to_string(),
            timeout_ms: 5000,
            trim: false,
            fallback_on_empty: true,
            streaming: false,
            edit_list: false,
        };
        let processor = PostProcessor::new(&config);
        let result = processor.process("hello world.").await;
        assert_eq!(result, "hello world. ");
    }

    #[tokio::test]
    async fn test_no_trim_still_strips_trailing_newlines() {
        // Even with trim = false, trailing newlines (shell artifacts) are stripped
        let config = PostProcessConfig {
            command: "echo 'hello'".to_string(),
            timeout_ms: 5000,
            trim: false,
            fallback_on_empty: true,
            streaming: false,
            edit_list: false,
        };
        let processor = PostProcessor::new(&config);
        let result = processor.process("ignored").await;
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn test_no_fallback_on_empty_returns_empty() {
        // When fallback_on_empty = false, empty output is returned as-is
        let config = PostProcessConfig {
            command: "printf ''".to_string(),
            timeout_ms: 5000,
            trim: true,
            fallback_on_empty: false,
            streaming: false,
            edit_list: false,
        };
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn test_fallback_on_empty_default_returns_original() {
        // Default behavior: empty output falls back to original text
        let config = make_config("printf ''", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "original text");
    }

    #[tokio::test]
    async fn test_no_trim_no_fallback_combination() {
        // Both options off: whatever the command emits is returned verbatim,
        // empty included.
        let config = PostProcessConfig {
            command: "printf ''".to_string(),
            timeout_ms: 5000,
            trim: false,
            fallback_on_empty: false,
            streaming: false,
            edit_list: false,
        };
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn test_trim_then_empty_triggers_fallback() {
        // Whitespace-only output should be considered empty after trimming,
        // and trigger the fallback when fallback_on_empty is on.
        let config = PostProcessConfig {
            command: "printf '   \\n  '".to_string(),
            timeout_ms: 5000,
            trim: true,
            fallback_on_empty: true,
            streaming: false,
            edit_list: false,
        };
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "original text");
    }

    #[tokio::test]
    async fn test_trim_then_empty_no_fallback_returns_empty() {
        // Same scenario but fallback off: empty string surfaces.
        let config = PostProcessConfig {
            command: "printf '   \\n  '".to_string(),
            timeout_ms: 5000,
            trim: true,
            fallback_on_empty: false,
            streaming: false,
            edit_list: false,
        };
        let processor = PostProcessor::new(&config);
        let result = processor.process("original text").await;
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn test_context_passed_via_env_var() {
        // Command prints VOXTYPE_CONTEXT env var, stdin is current text
        let config = make_config("echo \"context:$VOXTYPE_CONTEXT stdin:$(cat)\"", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor
            .process_with_context("current text", Some("previous text"))
            .await;
        assert_eq!(result, "context:previous text stdin:current text");
    }

    #[tokio::test]
    async fn test_no_context_env_var_when_none() {
        // VOXTYPE_CONTEXT should not be set when context is None
        let config = make_config(
            "echo \"context:${VOXTYPE_CONTEXT:-unset} stdin:$(cat)\"",
            5000,
        );
        let processor = PostProcessor::new(&config);
        let result = processor.process_with_context("current text", None).await;
        assert_eq!(result, "context:unset stdin:current text");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_context_env_not_inherited_from_parent() {
        // Even if VOXTYPE_CONTEXT is set in parent env, it should be cleared when context is None.
        // Uses current_thread runtime because std::env::set_var is not thread-safe
        // and will become unsafe in Rust edition 2024.
        std::env::set_var("VOXTYPE_CONTEXT", "stale parent context");
        let config = make_config("echo \"${VOXTYPE_CONTEXT:-unset}\"", 5000);
        let processor = PostProcessor::new(&config);
        let result = processor.process_with_context("text", None).await;
        std::env::remove_var("VOXTYPE_CONTEXT");
        assert_eq!(result, "unset");
    }

    async fn collect_stream(
        command: &str,
        timeout_ms: u64,
    ) -> (Vec<(String, u128)>, super::StreamEnd) {
        let mut config = make_config(command, timeout_ms);
        config.streaming = true;
        let processor = PostProcessor::new(&config);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let started = std::time::Instant::now();
        let run = processor.process_streaming("in", None, tx);
        tokio::pin!(run);
        let mut got = Vec::new();
        let mut end = None;
        loop {
            tokio::select! {
                p = rx.recv() => match p {
                    Some(p) => got.push((p, started.elapsed().as_millis())),
                    None => break,
                },
                e = &mut run, if end.is_none() => end = Some(e),
            }
        }
        (got, end.expect("run finished"))
    }

    #[tokio::test]
    async fn streaming_delivers_pieces_while_the_command_runs() {
        let (got, end) = collect_stream("printf 'One.'; sleep 0.4; printf ' Two.'", 5000).await;
        assert!(matches!(end, super::StreamEnd::Done));
        let text: String = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(text, "One. Two.");
        assert!(got.len() >= 2, "{got:?}");
        assert!(got[0].1 < 300, "first piece arrived after {}ms", got[0].1);
    }

    #[tokio::test]
    async fn streaming_tells_the_command_and_passes_stdin() {
        let (got, _) = collect_stream("printf '%s:' \"$VOXTYPE_STREAM_OUTPUT\"; cat", 5000).await;
        let text: String = got.into_iter().map(|(p, _)| p).collect();
        assert_eq!(text, "1:in");
    }

    #[tokio::test]
    async fn streaming_keeps_a_split_character_whole() {
        // "é" is 0xC3 0xA9; the two bytes arrive in separate writes.
        let (got, end) =
            collect_stream("printf 'caf\\303'; sleep 0.2; printf '\\251!'", 5000).await;
        assert!(matches!(end, super::StreamEnd::Done));
        let text: String = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(text, "café!");
    }

    #[tokio::test]
    async fn streaming_reports_failure_after_output() {
        let (got, end) = collect_stream("printf 'Part.'; exit 3", 5000).await;
        assert_eq!(got.len(), 1);
        assert!(matches!(
            end,
            super::StreamEnd::Failed(PostProcessError::NonZeroExit { code: Some(3), .. })
        ));
    }

    #[tokio::test]
    async fn streaming_times_out_and_keeps_what_arrived() {
        let (got, end) = collect_stream("printf 'Part.'; sleep 5; printf 'never'", 400).await;
        let text: String = got.into_iter().map(|(p, _)| p).collect();
        assert_eq!(text, "Part.");
        assert!(matches!(end, super::StreamEnd::Failed(PostProcessError::Timeout(_))));
    }

    #[tokio::test]
    async fn batch_runs_do_not_see_the_streaming_flag() {
        let config = make_config("printf '%s' \"${VOXTYPE_STREAM_OUTPUT:-unset}\"", 5000);
        let processor = PostProcessor::new(&config);
        assert_eq!(processor.process("x").await, "unset");
    }

}
