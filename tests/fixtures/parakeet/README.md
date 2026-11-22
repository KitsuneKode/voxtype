# Parakeet test audio

16-bit PCM, mono, 16 kHz.

| File | Source | Reference |
|------|--------|-----------|
| kaliko_hesitated.wav | LibriSpeech dev-clean `1272-135031-0016` | KALIKO HESITATED |

LibriSpeech (Panayotov et al., 2015) is licensed CC BY 4.0:
https://www.openslr.org/12

Greedy decoding with `parakeet-unified-en-0.6b` hears this as "Kaligo
hesitated." The vocabulary rescoring test expects "Kaliko" back with the
term in the vocabulary.
