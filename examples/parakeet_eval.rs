//! Run voxtype's own Parakeet final-pass path over WAV files, for
//! before/after accuracy and latency comparisons of `[parakeet]` options.
//!
//! The `[parakeet]` table is read from a TOML file, so any key the daemon
//! takes (`pad_secs`, `vocabulary_file`, ...) can be compared on the same
//! audio. Prints one JSON object per file: `{"file", "secs", "ms", "text"}`.
//!
//! ```text
//! ORT_DYLIB_PATH=.../libonnxruntime.so cargo run --release \
//!   --features parakeet,parakeet-load-dynamic,onnx-load-dynamic \
//!   --example parakeet_eval -- PARAKEET.toml FILE.wav...
//! ```

use std::time::Instant;
use voxtype::config::ParakeetConfig;
use voxtype::transcribe::Transcriber;

fn read_wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16000, "{path}: need 16 kHz");
    assert_eq!(spec.channels, 1, "{path}: need mono");
    r.samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: parakeet_eval PARAKEET.toml FILE.wav...");
        std::process::exit(2);
    }
    let toml_text = std::fs::read_to_string(&args[1]).expect("read config");
    let config: ParakeetConfig = toml::from_str(&toml_text).expect("parse [parakeet] table");

    let t = Instant::now();
    let transcriber: Box<dyn Transcriber> = if config.streaming {
        Box::new(
            voxtype::transcribe::parakeet_streaming::ParakeetStreamingTranscriber::new(&config)
                .unwrap_or_else(|e| panic!("load: {e}")),
        )
    } else {
        Box::new(
            voxtype::transcribe::parakeet::ParakeetTranscriber::new(&config)
                .unwrap_or_else(|e| panic!("load: {e}")),
        )
    };
    eprintln!("loaded in {}ms", t.elapsed().as_millis());

    for file in &args[2..] {
        let audio = read_wav(file);
        let secs = audio.len() as f64 / 16000.0;
        let t = Instant::now();
        let text = transcriber.transcribe(&audio).unwrap_or_else(|e| format!("<error: {e}>"));
        let ms = t.elapsed().as_millis();
        println!(
            "{}",
            serde_json::json!({ "file": file, "secs": (secs * 100.0).round() / 100.0, "ms": ms, "text": text })
        );
    }
}
