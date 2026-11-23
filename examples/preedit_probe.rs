//! Exercise `[output] preedit` against the running compositor, no daemon or audio.
//!
//! `preedit_probe` binds the input method for a second, reports whether the
//! focused field activated it, and releases it without sending any text.
//! `preedit_probe demo` shows a preedit in the focused field for two seconds
//! and removes it; `preedit_probe commit` then commits it instead.
//! Set `RUST_LOG=voxtype=debug` to see the compositor's events.
use std::time::Duration;
use voxtype::output::preedit::{CommitOutcome, PreeditHandle};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "voxtype=info".into()),
        )
        .init();
    let mode = std::env::args().nth(1).unwrap_or_default();
    let Some(preedit) = PreeditHandle::wayland() else {
        println!("PREEDIT_UNSUPPORTED_BUILD");
        return;
    };
    preedit.begin();
    let mut active = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if preedit.text_input_active() {
            active = true;
            break;
        }
    }
    println!("PREEDIT_BOUND text_input_active={active}");
    if active && (mode == "demo" || mode == "commit") {
        for text in ["voxtype", "voxtype preedit", "voxtype preedit test"] {
            preedit.set(text);
            tokio::time::sleep(Duration::from_millis(700)).await;
        }
        if mode == "commit" {
            let outcome = preedit.commit("voxtype preedit test").await;
            println!("PREEDIT_COMMIT {outcome:?}");
            assert_eq!(outcome, CommitOutcome::Committed);
        }
    }
    preedit.end_now().await;
    println!("PREEDIT_RELEASED");
}
