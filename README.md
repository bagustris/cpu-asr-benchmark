# cpu-asr-benchmark

Benchmark WER (Word Error Rate) and RTF (Real-Time Factor) for three
CPU-only ASR backends on your own test data.

| Backend | Library | Model |
|---------|---------|-------|
| `onnx` | [onnx-asr](https://github.com/yeyupiaoling/ONNX-ASR) + ONNX Runtime | NeMo Parakeet TDT 0.6B v2 |
| `sherpa` | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Zipformer streaming transducer (EN, 2023-06-26), Parakeet TDT V2 EN |
| `whisper` | [faster-whisper](https://github.com/guillaumekln/faster-whisper) (CTranslate2, int8) | Whisper small (or any size) |

## Metrics

### WER — Word Error Rate

Computed from scratch using the [`editdistance`](https://pypi.org/project/editdistance/) package,
following the approach in [speechain](https://github.com/human-ai-lab/speechain):

```
WER = Σ edit_distance(hyp_words, ref_words) / Σ len(ref_words)
```

Both hypothesis and reference are normalised (lowercased, punctuation stripped) before
tokenising on whitespace.

### RTF — Real-Time Factor

```
RTF = processing_time / audio_duration
```

- RTF < 1 → faster than real-time (good for live ASR)
- RTF > 1 → slower than real-time

## Setup

### 1. Install core dependencies

```bash
pip install editdistance soundfile numpy
```

### 2. Install backends you want to test

```bash
# ONNX backend
pip install onnx-asr onnxruntime

# Sherpa-ONNX backend
pip install sherpa-onnx

# Whisper backend
pip install faster-whisper
```

### 3. Prepare a manifest (optional)  

Create a TSV file with audio paths and reference transcripts:

```tsv
# audio_path	reference_text
data/audio/utt001.wav	hello world
data/audio/utt002.wav	the quick brown fox
```

For LibriSpeech, use the helper script:

```bash
python scripts/make_manifest_librispeech.py \
    --data-dir /path/to/LibriSpeech/test-clean \
    --output data/test-clean.tsv
```

## Usage

```bash
# Benchmark all three backends
python benchmark.py --manifest data/test-clean.tsv

# Specific backends only
python benchmark.py --manifest data/test-clean.tsv --backends onnx whisper

# Custom model and thread settings
python benchmark.py --manifest data/test-clean.tsv \
    --backends whisper \
    --whisper-model base \
    --whisper-threads 8 \
    --whisper-language en

# Sherpa with custom model directory
python benchmark.py --manifest data/test-clean.tsv \
    --backends sherpa \
    --sherpa-model-dir /path/to/sherpa-onnx-model \
    --sherpa-model-type model-type-name

# Save full results to JSON
python benchmark.py --manifest data/test-clean.tsv \
    --output results.json

# Verbose: show REF/HYP for each utterance
python benchmark.py --manifest data/test-clean.tsv --verbose 

# Direclyly specify audio and reference pairs without a manifest
python3 benchmark.py \
      --data-dir /data/LibriSpeech/dev-clean-2 \
      --backends sherpa \
      --sherpa-model-dir /home/bagus/github/live-asr-sherpa/src/model-parakeet \
      --output results_sherpa_parakeet.json

```

## Output  
```
============================================================
  Backend: ONNX
============================================================
  Loading model … done (3.2s)
  [   1/100] ✓  RTF=0.182  WER=  0.0%
  [   2/100] ✗  RTF=0.201  WER= 14.3%
  ...
  ── Aggregate (onnx) ──
  Utterances : 100
  WER        : 4.21%
  Mean RTF   : 0.197
  Audio total: 523.4s
  Proc total : 103.2s

============================================================
  SUMMARY
============================================================
  Backend                  WER (%)   RTF       #Utts
  ---------------------------------------------------------
  onnx-asr (parakeet)      1.84      0.1559    1089
  sherpa-zipformer         3.18      0.0533    1089
  whisper                  4.71      0.2767    1089
  sherpa-parakeet fp16     1.84      0.0834    1089
  sherpa-parakeet int8     1.97      0.0415    1089
============================================================
```

## Project Structure

```
cpu-asr-benchmark/
├── benchmark.py            # Main CLI entry point
├── metrics.py              # WER and RTF calculation (editdistance-based)
├── requirements.txt
├── backends/
│   ├── __init__.py
│   ├── base.py             # Abstract ASRBackend interface
│   ├── onnx_backend.py     # onnx-asr / ONNX Runtime
│   ├── sherpa_backend.py   # sherpa-onnx streaming transducer
│   └── whisper_backend.py  # faster-whisper (CTranslate2)
├── data/
│   └── README.md           # Data format and dataset suggestions
└── scripts/
    └── make_manifest_librispeech.py
```

## Notes

- **Sherpa RTF includes simulated streaming overhead.** The sherpa backend feeds
  audio in small chunks (default 0.1 s) to replicate online decoding, so its RTF
  reflects streaming latency, not just pure decode speed.
- **Whisper warmup** is performed during model load and is excluded from RTF
  measurements.
- **Audio resampling**: 16 kHz mono WAV is preferred. Other formats are supported
  via `soundfile`; resampling requires `resampy`.
