# Test Data

Place your benchmark audio files here and create a manifest file.

## Manifest Format

A manifest is a tab-separated (`.tsv`) or comma-separated (`.csv`) file with
two columns:

```
audio_path    reference_text
```

- `audio_path`: path to a WAV file (relative paths resolve from the manifest's directory)
- `reference_text`: ground-truth transcript for WER calculation
- Lines starting with `#` are treated as comments

### Example (`test.tsv`)

```tsv
audio/utt001.wav	hello world
audio/utt002.wav	the quick brown fox jumps over the lazy dog
audio/utt003.wav	speech recognition is hard but fun
```

## Recommended Datasets

| Dataset | Link | Notes |
|---------|------|-------|
| LibriSpeech test-clean | openslr.org/12 | 2620 utts, clean read speech |
| LibriSpeech test-other | openslr.org/12 | 2939 utts, challenging acoustics |
| Common Voice (English) | commonvoice.mozilla.org | Various accents |

## Audio Requirements

- **Format**: WAV (16-bit PCM recommended; other formats supported via soundfile)
- **Sample rate**: 16000 Hz preferred; other rates are resampled automatically if `resampy` is installed
- **Channels**: mono preferred; stereo is converted to mono by averaging channels

## Generating a LibriSpeech Manifest

```bash
# After downloading and extracting LibriSpeech test-clean:
python ../scripts/make_manifest_librispeech.py \
    --data-dir /path/to/LibriSpeech/test-clean \
    --output test-clean.tsv
```
