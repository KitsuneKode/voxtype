//! Replay a dictation take into the focused field through the daemon's own
//! delivery code, for measuring how an app takes it. Run it only against a
//! compositor of your own (a headless sway), never the live session.
//!
//! `preedit_replay <take.txt> [final.txt]` grows the preedit a few words at a
//! time the way a streaming take does, then delivers the final text the way
//! the daemon does for a long take: the preedit shows the transcript while the
//! cleanup runs, then [`PreeditOutput`] commits it, or the paste chain takes it
//! when the commit is not possible.
//!
//! Environment: `TICK_MS` (320), `WORDS_PER_TICK` (1), `PP_MS` (800),
//! `DELIVERY` (`preedit` or `paste`).
use std::time::{Duration, Instant};
use voxtype::config::{OutputConfig, OutputMode};
use voxtype::output::preedit::{PreeditHandle, PreeditOutput};
use voxtype::output::TextOutput;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "voxtype=info".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let take = std::fs::read_to_string(args.next().expect("take.txt")).expect("read take");
    let take = take.trim().to_string();
    let final_text = args
        .next()
        .map(|p| std::fs::read_to_string(p).expect("read final").trim().to_string())
        .unwrap_or_else(|| take.clone());
    let tick = Duration::from_millis(env_u64("TICK_MS", 320));
    let per_tick = env_u64("WORDS_PER_TICK", 1) as usize;
    let pp = Duration::from_millis(env_u64("PP_MS", 800));
    let delivery = std::env::var("DELIVERY").unwrap_or_else(|_| "preedit".into());
    // Experiments: show only the last CAP_CHARS of the take as preedit, or
    // commit all but the last KEEP_WORDS words as the take goes.
    let cap = env_u64("CAP_CHARS", 0) as usize;
    let keep = env_u64("KEEP_WORDS", 0) as usize;
    let mut committed = 0usize;

    let preedit = PreeditHandle::wayland().expect("preedit feature");
    preedit.begin();
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if preedit.text_input_active() {
            break;
        }
    }
    println!("REPLAY active={}", preedit.text_input_active());
    let start = Instant::now();
    let stamp = |what: &str| println!("REPLAY {:>8.1} {what}", start.elapsed().as_secs_f64() * 1000.0);

    let words: Vec<&str> = take.split_whitespace().collect();
    let mut n = 0;
    while n < words.len() {
        n = (n + per_tick).min(words.len());
        if keep > 0 && n > committed + keep {
            let upto = n - keep;
            let mut piece = words[committed..upto].join(" ");
            piece.push(' ');
            preedit.set(&words[upto..n].join(" "));
            let _ = preedit.commit(&piece).await;
            committed = upto;
        }
        let shown = words[committed..n].join(" ");
        if cap > 0 && shown.len() > cap {
            let mut at = shown.len() - cap;
            while !shown.is_char_boundary(at) {
                at += 1;
            }
            let tail = &shown[at..];
            let tail = tail.split_once(' ').map_or(tail, |(_, t)| t);
            preedit.set(&format!("\u{2026}{tail}"));
        } else {
            preedit.set(&shown);
        }
        tokio::time::sleep(tick).await;
    }
    stamp(&format!("stop words={} chars={}", words.len(), take.chars().count()));
    // The daemon shows the transcript while the cleanup runs.
    if keep == 0 && cap == 0 {
        preedit.set(&take);
    }
    if std::env::var("STREAM_PIECES").is_ok_and(|v| v == "1") {
        // What `stream_post_process` does with a preedit: each cleaned
        // sentence replaces the preedit with the cleaned text so far.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut shown = String::new();
        for piece in final_text.split_inclusive(". ") {
            shown.push_str(piece);
            preedit.set(shown.trim());
            stamp(&format!("piece shown={}", shown.trim().len()));
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
    } else {
        tokio::time::sleep(pp).await;
    }
    stamp("cleanup done");

    let mut config = OutputConfig::default();
    if delivery == "paste" {
        config.mode = OutputMode::Paste;
        preedit.clear_now().await;
        stamp("preedit cleared");
        let chain = voxtype::output::create_output_chain(&config);
        for out in &chain {
            if out.output(&final_text).await.is_ok() {
                stamp(&format!("delivered via {}", out.name()));
                break;
            }
        }
    } else {
        let out = PreeditOutput::new(preedit.clone(), &config);
        let rest = if keep > 0 { words[committed..].join(" ") } else { final_text.clone() };
        let r = out.output(&rest).await;
        stamp(&format!("delivered via preedit: {r:?}"));
    }
    preedit.release_now().await;
    stamp("released");
}
