//! Time the daemon's final pass (`ParakeetStreamingTranscriber::transcribe`)
//! on WAV files, with and without vocabulary rescoring, on whatever execution
//! provider the build's features select (CUDA with `parakeet-cuda`).
//!
//! ```text
//! cargo run --release --features parakeet-cuda,parakeet-load-dynamic,onnx-load-dynamic \
//!   --example final_pass_bench -- MODEL_DIR [VOCABULARY [CORRECTIONS]] -- FILE.wav...
//! ```
//!
//! Each file is decoded three times per setting; `RUST_LOG=voxtype=debug`
//! shows the split between the encoder and the rescoring.

use std::time::Instant;
use voxtype::config::ParakeetConfig;
use voxtype::transcribe::parakeet_streaming::ParakeetStreamingTranscriber;
use voxtype::transcribe::Transcriber;

fn read_wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    assert_eq!(r.spec().sample_rate, 16000, "{path}: need 16 kHz");
    r.samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect()
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args
        .iter()
        .position(|a| a == "--")
        .expect("-- before the files");
    let (head, files) = (&args[..split], &args[split + 1..]);
    let model = head.first().expect("MODEL_DIR").clone();
    let base = ParakeetConfig {
        model,
        streaming: true,
        final_pass_segment_secs: 45.0,
        ..ParakeetConfig::default()
    };
    let mut settings = vec![("plain", base.clone())];
    if let Some(vocabulary) = head.get(1) {
        settings.push((
            "vocabulary",
            ParakeetConfig {
                vocabulary_file: Some(vocabulary.into()),
                corrections_file: head.get(2).map(Into::into),
                ..base.clone()
            },
        ));
    }
    for (name, config) in settings {
        let t = Instant::now();
        let transcriber = ParakeetStreamingTranscriber::new(&config).expect("load");
        eprintln!("{name}: loaded in {}ms", t.elapsed().as_millis());
        for file in files {
            let audio = read_wav(file);
            let secs = audio.len() as f32 / 16000.0;
            for run in 0..4 {
                // A different length every run, as every real take has.
                let audio = &audio[..audio.len() - run * 1601];
                let t = Instant::now();
                let text = match transcriber.transcribe(audio) {
                    Ok(text) => text,
                    Err(e) => {
                        // The daemon's retry: short pieces, cut at pauses.
                        let failed = t.elapsed().as_millis();
                        let retry = voxtype::transcribe::segment::transcribe_in_pieces(
                            &transcriber,
                            audio,
                            16000,
                            10.0,
                        );
                        println!(
                            "{name}\t{file}\t{secs:.1}s\trun {run}\twhole take failed after {failed}ms, pieces {} after {}ms ({} chars): {}",
                            if retry.is_ok() { "ok" } else { "failed" },
                            t.elapsed().as_millis(),
                            retry.as_ref().map_or(0, |t| t.chars().count()),
                            e.to_string().chars().take(60).collect::<String>()
                        );
                        continue;
                    }
                };
                println!(
                    "{name}\t{file}\t{secs:.1}s\trun {run}\t{}ms\t{} chars",
                    t.elapsed().as_millis(),
                    text.chars().count()
                );
            }
        }
    }
}
