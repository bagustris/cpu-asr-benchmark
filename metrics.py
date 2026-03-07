"""
WER and RTF metrics — computed from scratch using the editdistance package.

WER calculation mirrors the approach in speechain/criterion/error_rate.py:
  - tokenize both hypothesis and reference by splitting on whitespace
  - compute edit distance (insertions + deletions + substitutions)
  - WER = total_edit_distance / total_reference_words

RTF (Real-Time Factor) = processing_time / audio_duration
  RTF < 1  → faster than real-time (good for live ASR)
  RTF > 1  → slower than real-time
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List

import editdistance


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation to get a canonical word sequence."""
    text = text.lower()
    # Remove punctuation except apostrophes inside words (e.g. "don't")
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Per-utterance metrics
# ---------------------------------------------------------------------------

@dataclass
class UtteranceResult:
    audio_path: str
    reference: str
    hypothesis: str
    audio_duration: float       # seconds
    processing_time: float      # seconds

    # Filled in by compute()
    ref_words: List[str] = field(default_factory=list)
    hyp_words: List[str] = field(default_factory=list)
    edit_distance: int = 0
    wer: float = 0.0
    rtf: float = 0.0

    def compute(self) -> "UtteranceResult":
        ref_norm = normalize_text(self.reference)
        hyp_norm = normalize_text(self.hypothesis)

        self.ref_words = ref_norm.split()
        self.hyp_words = hyp_norm.split()

        # editdistance.eval returns the Levenshtein distance between two sequences
        self.edit_distance = editdistance.eval(self.hyp_words, self.ref_words)

        ref_len = len(self.ref_words)
        self.wer = (self.edit_distance / ref_len) if ref_len > 0 else 0.0
        self.rtf = (
            self.processing_time / self.audio_duration
            if self.audio_duration > 0
            else float("inf")
        )
        return self


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Corpus-level WER and mean RTF computed over a list of utterance results."""

    total_edit_distance: int = 0
    total_ref_words: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    n_utterances: int = 0

    @property
    def wer(self) -> float:
        """Corpus-level WER: sum(edit_dist) / sum(ref_words)."""
        return (
            self.total_edit_distance / self.total_ref_words
            if self.total_ref_words > 0
            else 0.0
        )

    @property
    def wer_pct(self) -> float:
        return self.wer * 100

    @property
    def mean_rtf(self) -> float:
        """Average per-utterance RTF."""
        return (
            self.total_processing_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else float("inf")
        )

    @classmethod
    def from_results(cls, results: List[UtteranceResult]) -> "AggregateMetrics":
        m = cls()
        for r in results:
            m.total_edit_distance += r.edit_distance
            m.total_ref_words += len(r.ref_words)
            m.total_audio_duration += r.audio_duration
            m.total_processing_time += r.processing_time
            m.n_utterances += 1
        return m
