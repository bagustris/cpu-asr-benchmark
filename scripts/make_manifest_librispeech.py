#!/usr/bin/env python3
"""
Generate a TSV manifest from a LibriSpeech directory.

Usage:
  python make_manifest_librispeech.py \\
      --data-dir /path/to/LibriSpeech/test-clean \\
      --output ../data/test-clean.tsv
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create benchmark manifest from LibriSpeech")
    p.add_argument("--data-dir", required=True, help="LibriSpeech split directory")
    p.add_argument("--output", required=True, help="Output TSV manifest path")
    p.add_argument("--max-utts", type=int, default=None,
                   help="Limit to first N utterances (useful for quick tests)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    # LibriSpeech structure: <speaker>/<chapter>/<utt>.flac + <utt>.trans.txt
    for trans_file in sorted(data_dir.rglob("*.trans.txt")):
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
                # Try both .flac and .wav
                for ext in (".flac", ".wav"):
                    audio_path = chapter_dir / f"{utt_id}{ext}"
                    if audio_path.exists():
                        records.append((str(audio_path), text.lower()))
                        break

    if args.max_utts:
        records = records[: args.max_utts]

    with open(output, "w", encoding="utf-8") as f:
        f.write("# audio_path\treference_text\n")
        for audio_path, text in records:
            f.write(f"{audio_path}\t{text}\n")

    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
