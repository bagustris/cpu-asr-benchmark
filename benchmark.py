#!/usr/bin/env python3
"""
cpu-asr-benchmark — compare WER and RTF across three CPU ASR backends.

Supported backends:
  onnx    → NeMo Parakeet TDT via onnx-asr
  sherpa  → Zipformer transducer via sherpa-onnx
  whisper → faster-whisper (CTranslate2, int8)

Input — one of:
  --data-dir PATH   LibriSpeech split directory (auto-builds manifest)
  --manifest FILE   Pre-built TSV/CSV: audio_path <sep> reference_text

Example
-------
  # Run all backends on LibriSpeech dev-clean-2
  python benchmark.py --data-dir /data/LibriSpeech/dev-clean-2

  # Only whisper, save JSON results
  python benchmark.py --data-dir /data/LibriSpeech/dev-clean-2 \\
      --backends whisper --whisper-model base --output results.json

  # Use a pre-built manifest
  python benchmark.py --manifest data/test-clean.tsv --backends onnx sherpa
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from metrics import AggregateMetrics, UtteranceResult
from backends import OnnxBackend, SherpaBackend, WhisperBackend
from backends.base import ASRBackend

# Default paths
DEFAULT_DATA_DIR = "/data/LibriSpeech/dev-clean-2"
DEFAULT_SHERPA_MODEL = str(
    Path(__file__).resolve().parent.parent / "live-asr-sherpa" / "src" / "model"
)


# ---------------------------------------------------------------------------
# Manifest generation from LibriSpeech directory
# ---------------------------------------------------------------------------

def manifest_from_librispeech(data_dir: str) -> List[Tuple[str, str]]:
    """
    Walk a LibriSpeech split directory and return (audio_path, reference) pairs.

    Structure expected:
      <data_dir>/<speaker>/<chapter>/<utt_id>.flac
      <data_dir>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

    References are lowercased to match normalised ASR output.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    records: List[Tuple[str, str]] = []
    for trans_file in sorted(data_path.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                for ext in (".flac", ".wav"):
                    audio_path = chapter_dir / f"{utt_id}{ext}"
                    if audio_path.exists():
                        records.append((str(audio_path), text.lower()))
                        break

    if not records:
        print(f"Error: no audio files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    return records


# ---------------------------------------------------------------------------
# Manifest parsing from TSV/CSV file
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Tuple[str, str]]:
    """Parse a TSV or CSV manifest into (audio_path, reference_text) pairs."""
    records: List[Tuple[str, str]] = []
    manifest_dir = Path(path).parent

    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            audio_path = row[0].strip()
            reference = row[1].strip()
            if not Path(audio_path).is_absolute():
                audio_path = str(manifest_dir / audio_path)
            records.append((audio_path, reference))

    if not records:
        print(f"Error: no valid records in manifest: {path}", file=sys.stderr)
        sys.exit(1)
    return records


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """Load audio file, convert to mono float32, resample if needed."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)

    duration = len(audio) / target_sr
    return audio, duration


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_backend(
    backend: ASRBackend,
    records: List[Tuple[str, str]],
    sample_rate: int = 16000,
    verbose: bool = False,
) -> Tuple[List[UtteranceResult], AggregateMetrics]:
    """Run all utterances through one backend; return per-utterance and aggregate results."""
    results: List[UtteranceResult] = []

    print(f"\n{'='*60}")
    print(f"  Backend: {backend.name.upper()}")
    print(f"{'='*60}")
    print("  Loading model … ", end="", flush=True)
    t0 = time.monotonic()
    backend.load()
    load_time = time.monotonic() - t0
    print(f"done ({load_time:.1f}s)")

    for i, (audio_path, reference) in enumerate(records, 1):
        try:
            audio, duration = load_audio(audio_path, target_sr=sample_rate)
        except Exception as exc:
            print(f"  [{i:4d}] SKIP {Path(audio_path).name}: {exc}", flush=True)
            continue

        t_start = time.monotonic()
        try:
            hypothesis = backend.transcribe(audio, sample_rate=sample_rate)
        except Exception as exc:
            print(f"  [{i:4d}] ERROR {Path(audio_path).name}: {exc}", flush=True)
            hypothesis = ""
        proc_time = time.monotonic() - t_start

        utt = UtteranceResult(
            audio_path=audio_path,
            reference=reference,
            hypothesis=hypothesis,
            audio_duration=duration,
            processing_time=proc_time,
        ).compute()

        results.append(utt)

        if verbose:
            print(
                f"  [{i:4d}/{len(records)}] RTF={utt.rtf:.3f}  WER={utt.wer*100:6.1f}%"
            )
            print(f"    REF: {reference[:80]}")
            print(f"    HYP: {hypothesis[:80]}")
        else:
            marker = "✓" if utt.wer == 0.0 else "✗"
            print(
                f"  [{i:4d}/{len(records)}] {marker}  "
                f"RTF={utt.rtf:.3f}  WER={utt.wer*100:5.1f}%",
                flush=True,
            )

    agg = AggregateMetrics.from_results(results)
    print(f"\n  ── Aggregate ({backend.name}) ──")
    print(f"  Utterances : {agg.n_utterances}")
    print(f"  WER        : {agg.wer_pct:.2f}%")
    print(f"  Mean RTF   : {agg.mean_rtf:.4f}")
    print(f"  Audio total: {agg.total_audio_duration:.1f}s")
    print(f"  Proc total : {agg.total_processing_time:.1f}s")

    return results, agg


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Backend':<12}  {'WER (%)':>10}  {'Mean RTF':>10}  {'#Utts':>6}")
    print("  " + "-" * 46)
    for name, agg in summary.items():
        print(
            f"  {name:<12}  {agg.wer_pct:>10.2f}  {agg.mean_rtf:>10.4f}  {agg.n_utterances:>6}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark WER and RTF for CPU ASR backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        metavar="PATH",
        help="LibriSpeech split directory (auto-generates manifest)",
    )
    src.add_argument(
        "--manifest",
        metavar="FILE",
        help="Pre-built TSV/CSV manifest: audio_path <sep> reference_text",
    )

    p.add_argument(
        "--backends", nargs="+", choices=["onnx", "sherpa", "whisper"],
        default=["onnx", "sherpa", "whisper"],
        help="Which backends to benchmark",
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--threads", type=int, default=4,
                   help="CPU threads for all backends (overridden by backend-specific flags)")
    p.add_argument("--max-utts", type=int, default=None,
                   help="Limit to first N utterances (for quick testing)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-utterance REF/HYP lines")
    p.add_argument("--output", metavar="FILE",
                   help="Save full results to JSON file")

    # ONNX backend
    g = p.add_argument_group("onnx backend")
    g.add_argument("--onnx-model", default="nemo-parakeet-tdt-0.6b-v2",
                   help="onnx-asr model name or local ONNX path")
    g.add_argument("--onnx-threads", type=int, default=None)

    # Sherpa backend
    g = p.add_argument_group("sherpa backend")
    g.add_argument("--sherpa-model-dir", default=DEFAULT_SHERPA_MODEL,
                   help="Directory with Sherpa-ONNX model files")
    g.add_argument("--sherpa-threads", type=int, default=None)
    g.add_argument("--sherpa-chunk-size", type=float, default=0.1,
                   help="Simulated streaming chunk size in seconds")
    g.add_argument("--sherpa-model-type", default="online",
                   choices=["online", "nemo_transducer"],
                   help="Model type: 'online' for Zipformer streaming, 'nemo_transducer' for NeMo Parakeet offline")

    # Whisper backend
    g = p.add_argument_group("whisper backend")
    g.add_argument("--whisper-model", default="small",
                   help="Model name (tiny/base/small/medium/large-v3) or local path")
    g.add_argument("--whisper-threads", type=int, default=None)
    g.add_argument("--whisper-language", default="en",
                   help="Language code or 'auto' for detection")
    g.add_argument("--whisper-beam-size", type=int, default=5)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load records
    if args.manifest:
        records = load_manifest(args.manifest)
        print(f"Loaded {len(records)} utterances from {args.manifest}")
    else:
        records = manifest_from_librispeech(args.data_dir)
        print(f"Loaded {len(records)} utterances from {args.data_dir}")

    if args.max_utts:
        records = records[: args.max_utts]
        print(f"Limited to first {len(records)} utterances (--max-utts)")

    # Build requested backends
    backend_map: dict[str, ASRBackend] = {}
    if "onnx" in args.backends:
        backend_map["onnx"] = OnnxBackend(
            model_name=args.onnx_model,
            cpu_threads=args.onnx_threads or args.threads,
        )
    if "sherpa" in args.backends:
        backend_map["sherpa"] = SherpaBackend(
            model_dir=args.sherpa_model_dir,
            num_threads=args.sherpa_threads or args.threads,
            sample_rate=args.sample_rate,
            chunk_size=args.sherpa_chunk_size,
            model_type=args.sherpa_model_type,
        )
    if "whisper" in args.backends:
        language = None if args.whisper_language == "auto" else args.whisper_language
        backend_map["whisper"] = WhisperBackend(
            model_name=args.whisper_model,
            threads=args.whisper_threads or args.threads,
            language=language,
            beam_size=args.whisper_beam_size,
        )

    all_results: dict = {}
    summary: dict[str, AggregateMetrics] = {}

    for name, backend in backend_map.items():
        results, agg = run_backend(
            backend, records,
            sample_rate=args.sample_rate,
            verbose=args.verbose,
        )
        summary[name] = agg
        all_results[name] = {
            "aggregate": {
                "wer_pct": agg.wer_pct,
                "mean_rtf": agg.mean_rtf,
                "total_audio_duration_s": agg.total_audio_duration,
                "total_processing_time_s": agg.total_processing_time,
                "n_utterances": agg.n_utterances,
            },
            "utterances": [
                {
                    "audio_path": r.audio_path,
                    "reference": r.reference,
                    "hypothesis": r.hypothesis,
                    "audio_duration_s": r.audio_duration,
                    "processing_time_s": r.processing_time,
                    "edit_distance": r.edit_distance,
                    "wer": r.wer,
                    "rtf": r.rtf,
                }
                for r in results
            ],
        }

    print_summary(summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
