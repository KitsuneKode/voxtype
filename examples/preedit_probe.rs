//! Exercise `[output] preedit` against the running compositor, no daemon or audio.
//!
//! `preedit_probe` binds the input method for a second, reports whether the
//! focused field activated it, and releases it without sending any text.
//! `preedit_probe field` reports how much text the focused field shares
//! (lengths only). `preedit_probe demo` shows a preedit in the focused field for two seconds
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
    if active && mode == "field" {
        // Lengths only: the field may hold anything.
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let path = std::path::Path::new(&std::env::var("XDG_RUNTIME_DIR").unwrap_or_default())
            .join("voxtype/field_text.json");
        match std::fs::read_to_string(&path) {
            Ok(body) => {
                let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let len = |k: &str| v[k].as_str().map_or(0, |s| s.chars().count());
                println!(
                    "PREEDIT_FIELD before={} after={}",
                    len("before"),
                    len("after")
                );
            }
            Err(_) => println!("PREEDIT_FIELD none (the field sent no surrounding text)"),
        }
    }
    preedit.release_now().await;
    println!("PREEDIT_RELEASED");
}
