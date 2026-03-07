"""
Faster-Whisper backend (CTranslate2, int8 quantization).

Corresponds to the live-asr-whisper project.
"""

from __future__ import annotations

import numpy as np

from .base import ASRBackend


class WhisperBackend(ASRBackend):
    name = "whisper"

    def __init__(
        self,
        model_name: str = "small",
        threads: int = 4,
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        self.model_name = model_name
        self.threads = threads
        self.language = language or None  # None → auto-detect
        self.beam_size = beam_size
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.threads,
        )
        # Warmup to avoid cold-start penalty in first measurement
        dummy = np.zeros(16000, dtype=np.float32)
        list(
            self._model.transcribe(dummy, beam_size=1, language=self.language)[0]
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) == 0:
            return ""
        segments, _ = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments)
