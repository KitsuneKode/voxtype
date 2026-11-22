//! Run voxtype's own Parakeet final-pass path over WAV files, for
//! before/after accuracy and latency comparisons of `[parakeet]` options.
//!
//! The `[parakeet]` table is read from a TOML file, so any key the daemon
//! takes (`pad_secs`, `vocabulary_file`, ...) can be compared on the same
//! audio. Prints one JSON object per file: `{"file", "secs", "ms", "text"}`.
//!
//! With `EVAL_STREAM=1` (streaming models) each file is fed to the live
//! stream instead, as fast as the stream takes it, and the line also carries
//! the streamed text and whether the stop would have accepted it without a
//! final pass (`[parakeet] streaming_final_pass_skip_when_clean`).
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

    if std::env::var_os("EVAL_STREAM").is_some() {
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        for file in &args[2..] {
            let audio = read_wav(file);
            let line = runtime.block_on(stream_one(transcriber.as_ref(), &audio));
            println!(
                "{}",
                serde_json::json!({ "file": file, "secs": audio.len() as f64 / 16000.0, "stream": line.0, "clean": line.1 })
            );
        }
        return;
    }

    for file in &args[2..] {
        let audio = read_wav(file);
        let secs = audio.len() as f64 / 16000.0;
        let t = Instant::now();
        let text = transcriber
            .transcribe(&audio)
            .unwrap_or_else(|e| format!("<error: {e}>"));
        let ms = t.elapsed().as_millis();
        println!(
            "{}",
            serde_json::json!({ "file": file, "secs": (secs * 100.0).round() / 100.0, "ms": ms, "text": text })
        );
    }
}

/// Feed `audio` to a fresh stream in 20 ms blocks (the capture pump's size),
/// wait until the stream has taken all of it, and return the streamed text
/// and the stop's clean-acceptance verdict at that point.
async fn stream_one(transcriber: &dyn Transcriber, audio: &[f32]) -> (String, bool) {
    let streaming = transcriber.as_streaming().expect("model does not stream");
    let (tx, rx) = tokio::sync::mpsc::channel(1 << 16);
    let mut handle = streaming.start_stream(rx).expect("start stream");
    // Keep the event channel empty, as the daemon does; a full one stalls
    // the stream.
    let mut events = std::mem::replace(&mut handle.events, tokio::sync::mpsc::channel(1).1);
    let drain = tokio::spawn(async move { while events.recv().await.is_some() {} });
    for block in audio.chunks(320) {
        tx.send(block.to_vec()).await.expect("feed");
    }
    let progress = loop {
        let p = streaming.stream_progress().expect("stream progress");
        if p.fed >= audio.len() {
            break p;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    let clean = voxtype::transcribe::stream_is_clean(&progress, audio, 16000);
    let _ = std::mem::replace(&mut handle.cancel, tokio::sync::oneshot::channel().0).send(());
    drop(tx);
    let _ = drain.await;
    (progress.text.trim().to_string(), clean)
}
