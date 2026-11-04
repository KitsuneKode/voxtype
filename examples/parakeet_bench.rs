//! Benchmark the two Parakeet unified code paths the daemon runs on a
//! `streaming_commit = "on_stop"` + `streaming_final_pass` setup:
//!
//! - `offline`: the final pass (`ParakeetUnified::transcribe_audio`, one
//!   full-context encode of the whole recording).
//! - `stream`: the live preview (`transcribe_chunk` fed 20 ms blocks, as the
//!   capture pump does). Time is simulated: each block "arrives" at its audio
//!   timestamp and is processed once the previous block is done, so the run
//!   reports how far the stream falls behind real time (`drain_ms`: how long
//!   after the speech ends the stream would still be decoding).
//!
//! Prints one JSON object per file. Usage:
//!
//! ```text
//! ORT_DYLIB_PATH=.../libonnxruntime.so cargo run --release \
//!   --features parakeet,parakeet-load-dynamic,onnx-load-dynamic \
//!   --example parakeet_bench -- offline|stream MODEL_DIR THREADS \
//!   CHUNK LEFT RIGHT FILE.wav...
//! ```

use parakeet_rs::{ExecutionConfig, ParakeetUnified, ParakeetUnifiedHandle, UnifiedStreamingConfig};
use std::time::Instant;

fn read_wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16000, "{path}: need 16 kHz");
    assert_eq!(spec.channels, 1, "{path}: need mono");
    r.samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect()
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        eprintln!("usage: offline|stream MODEL_DIR THREADS CHUNK LEFT RIGHT FILE...");
        std::process::exit(2);
    }
    let mode = a[1].as_str();
    let threads: usize = a[3].parse().unwrap();
    let cfg = UnifiedStreamingConfig {
        chunk_secs: a[4].parse().unwrap(),
        left_context_secs: a[5].parse().unwrap(),
        right_context_secs: a[6].parse().unwrap(),
    }
    .validate()
    .expect("streaming config");

    let exec = ExecutionConfig::new().with_intra_threads(threads);
    let t = Instant::now();
    let handle = ParakeetUnifiedHandle::load(&a[2], Some(exec)).expect("load");
    eprintln!("loaded {} in {}ms (threads={threads})", a[2], t.elapsed().as_millis());

    for file in &a[7..] {
        let audio = read_wav(file);
        let secs = audio.len() as f64 / 16000.0;
        let mut m = ParakeetUnified::from_shared_with_streaming_config(&handle, cfg).unwrap();
        match mode {
            "offline" => {
                let t = Instant::now();
                let text = m.transcribe_audio(audio, 16000, 1).expect("offline");
                println!(
                    "{{\"file\":{},\"secs\":{secs:.2},\"ms\":{},\"text\":{}}}",
                    json_str(file),
                    t.elapsed().as_millis(),
                    json_str(text.trim())
                );
            }
            "offseg" => {
                // Final pass split at pauses (voxtype's segment planner).
                let max_secs: f32 = std::env::var("SEG_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0);
                let t = Instant::now();
                let mut parts = Vec::new();
                for (s, e) in voxtype::transcribe::segment::plan_segments(&audio, 16000, max_secs) {
                    let piece = m.transcribe_audio(audio[s..e].to_vec(), 16000, 1).expect("seg");
                    let piece = piece.trim().to_string();
                    if !piece.is_empty() {
                        parts.push(piece);
                    }
                }
                println!(
                    "{{\"file\":{},\"secs\":{secs:.2},\"ms\":{},\"text\":{}}}",
                    json_str(file),
                    t.elapsed().as_millis(),
                    json_str(&parts.join(" "))
                );
            }
            "stream" => {
                const BLOCK: usize = 320; // 20 ms, like the capture pump
                let mut clock = 0.0f64; // simulated wall time, seconds
                let mut busy = 0.0f64;
                let mut text = String::new();
                for (i, block) in audio.chunks(BLOCK).enumerate() {
                    let arrives = ((i + 1) * BLOCK) as f64 / 16000.0;
                    clock = clock.max(arrives);
                    let t = Instant::now();
                    text.push_str(&m.transcribe_chunk(block).expect("chunk"));
                    let d = t.elapsed().as_secs_f64();
                    busy += d;
                    clock += d;
                }
                let t = Instant::now();
                text.push_str(&m.flush().expect("flush"));
                let flush = t.elapsed().as_secs_f64();
                println!(
                    "{{\"file\":{},\"secs\":{secs:.2},\"busy_ms\":{:.0},\"drain_ms\":{:.0},\"flush_ms\":{:.0},\"text\":{}}}",
                    json_str(file),
                    busy * 1000.0,
                    (clock - secs).max(0.0) * 1000.0,
                    flush * 1000.0,
                    json_str(text.trim())
                );
            }
            _ => panic!("mode must be offline or stream"),
        }
    }
}
