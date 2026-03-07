"""
ONNX-ASR backend (NeMo Parakeet TDT via onnx-asr / ONNX Runtime).

Corresponds to the live-asr-onnx project.
Model: nemo-parakeet-tdt-0.6b-v2 (English, non-streaming)
"""

from __future__ import annotations

import os

import numpy as np

from .base import ASRBackend


class OnnxBackend(ASRBackend):
    name = "onnx"

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v2",
        cpu_threads: int = 4,
    ) -> None:
        self.model_name = model_name
        self.cpu_threads = cpu_threads
        self._model = None

    def load(self) -> None:
        import onnx_asr

        os.environ["OMP_NUM_THREADS"] = str(self.cpu_threads)
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        self._model = onnx_asr.load_model(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        return self._model.recognize(audio, sample_rate=sample_rate)
