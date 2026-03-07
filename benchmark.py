#!/usr/bin/env python3
"""
cpu-asr-benchmark — compare WER and RTF across three CPU ASR backends.

Supported backends:
  onnx    → NeMo Parakeet TDT via onnx-asr
  sherpa  → Zipformer transducer via sherpa-onnx
  whisper → faster-whisper (CTranslate2, int8)

Input manifest (--manifest): a TSV or CSV file with two columns:
  audio_path  <TAB or COMMA>  reference_text

Example
-------
  # Benchmark all backends on LibriSpeech test-clean manifest
  python benchmark.py --manifest data/test-clean.tsv --backends onnx sherpa whisper

  # Only run whisper with a specific model
  python benchmark.py --manifest data/test-clean.tsv --backends whisper \\
      --whisper-model base --whisper-threads 8

  # Run onnx and whisper, save results to JSON
  python benchmark.py --manifest data/test-clean.tsv --backends onnx whisper \\
      --output results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from metrics import AggregateMetrics, UtteranceResult
from backends import OnnxBackend, SherpaBackend, WhisperBackend
from backends.base import ASRBackend


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Load an audio file and resample to target_sr if needed.

    Returns
    -------
    audio    : float32 numpy array, shape (N,)
    duration : audio duration in seconds
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != target_sr:
        try:
            import resampy
            audio = resampy.resample(audio, sr, target_sr)
        except ImportError:
            raise RuntimeError(
                f"Audio '{path}' has sample rate {sr} Hz but target is {target_sr} Hz. "
                "Install resampy to enable resampling: pip install resampy"
            )

    duration = len(audio) / target_sr
    return audio, duration


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Tuple[str, str]]:
    """
    Parse a manifest file into (audio_path, reference_text) pairs.

    Accepts:
      - TSV  (tab-separated)
      - CSV  (comma-separated)
    First column = audio path, second column = reference text.
    Lines starting with '#' are treated as comments.
    """
    records: List[Tuple[str, str]] = []
    manifest_dir = Path(path).parent

    with open(path, newline="", encoding="utf-8") as f:
        # Auto-detect delimiter
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
            # Resolve relative paths relative to the manifest directory
            if not Path(audio_path).is_absolute():
                audio_path = str(manifest_dir / audio_path)
            records.append((audio_path, reference))

    if not records:
        raise ValueError(f"No valid records found in manifest: {path}")
    return records


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
    print(f"  Loading model … ", end="", flush=True)
    t0 = time.monotonic()
    backend.load()
    load_time = time.monotonic() - t0
    print(f"done ({load_time:.1f}s)")

    for i, (audio_path, reference) in enumerate(records, 1):
        try:
            audio, duration = load_audio(audio_path, target_sr=sample_rate)
        except Exception as exc:
            print(f"  [{i:4d}] SKIP {audio_path}: {exc}", flush=True)
            continue

        t_start = time.monotonic()
        try:
            hypothesis = backend.transcribe(audio, sample_rate=sample_rate)
        except Exception as exc:
            print(f"  [{i:4d}] ERROR {audio_path}: {exc}", flush=True)
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
                f"  [{i:4d}] RTF={utt.rtf:.3f}  WER={utt.wer*100:6.1f}%"
                f"  REF: {reference[:60]}"
            )
            print(f"         HYP: {hypothesis[:60]}")
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
    """Print a comparison table across all backends."""
    backends = list(summary.keys())
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    header = f"  {'Backend':<12}  {'WER (%)':>10}  {'Mean RTF':>10}  {'#Utts':>6}"
    print(header)
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
    p.add_argument(
        "--manifest", required=True,
        help="Path to TSV/CSV manifest: audio_path <sep> reference_text",
    )
    p.add_argument(
        "--backends", nargs="+", choices=["onnx", "sherpa", "whisper"],
        default=["onnx", "sherpa", "whisper"],
        help="Which backends to benchmark",
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--threads", type=int, default=4,
        help="CPU threads for all backends (overridden by backend-specific flags)",
    )
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-utterance REF/HYP lines")
    p.add_argument("--output", metavar="FILE",
                   help="Save results to JSON file")

    # ONNX backend options
    g_onnx = p.add_argument_group("onnx backend")
    g_onnx.add_argument("--onnx-model",
                        default="nemo-parakeet-tdt-0.6b-v2",
                        help="onnx-asr model name or local ONNX path")
    g_onnx.add_argument("--onnx-threads", type=int, default=None,
                        help="ONNX Runtime threads (defaults to --threads)")

    # Sherpa backend options
    g_sherpa = p.add_argument_group("sherpa backend")
    g_sherpa.add_argument("--sherpa-model-dir", default="model",
                          help="Directory containing Sherpa-ONNX model files")
    g_sherpa.add_argument("--sherpa-threads", type=int, default=None,
                          help="Sherpa-ONNX threads (defaults to --threads)")
    g_sherpa.add_argument("--sherpa-chunk-size", type=float, default=0.1,
                          help="Simulated streaming chunk size in seconds")

    # Whisper backend options
    g_whisper = p.add_argument_group("whisper backend")
    g_whisper.add_argument("--whisper-model", default="small",
                           help="Whisper model name (tiny/base/small/medium/large-v3)")
    g_whisper.add_argument("--whisper-threads", type=int, default=None,
                           help="faster-whisper threads (defaults to --threads)")
    g_whisper.add_argument("--whisper-language", default="en",
                           help="Language code or 'auto'")
    g_whisper.add_argument("--whisper-beam-size", type=int, default=5,
                           help="Beam size for decoding")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    print(f"Loaded {len(records)} utterances from {args.manifest}")

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
                "total_audio_duration": agg.total_audio_duration,
                "total_processing_time": agg.total_processing_time,
                "n_utterances": agg.n_utterances,
            },
            "utterances": [
                {
                    "audio_path": r.audio_path,
                    "reference": r.reference,
                    "hypothesis": r.hypothesis,
                    "audio_duration": r.audio_duration,
                    "processing_time": r.processing_time,
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
