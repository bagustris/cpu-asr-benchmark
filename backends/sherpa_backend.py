"""
Sherpa-ONNX backend (streaming Zipformer transducer).

Corresponds to the live-asr-sherpa project.
Model: sherpa-onnx-streaming-zipformer-en-2023-06-26

For benchmarking we simulate streaming by feeding fixed-size chunks to the
online recogniser and collecting the final result after all audio is consumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ASRBackend


class SherpaBackend(ASRBackend):
    name = "sherpa"

    def __init__(
        self,
        model_dir: str = "model",
        num_threads: int = 4,
        sample_rate: int = 16000,
        chunk_size: float = 0.1,  # seconds per simulated chunk
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_size)
        self._recognizer = None

    def _find(self, pattern: str) -> str:
        matches = sorted(self.model_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No file matching '{pattern}' in {self.model_dir}"
            )
        return str(matches[0])

    def load(self) -> None:
        import sherpa_onnx

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=self._find("tokens.txt"),
            encoder=self._find("encoder*.onnx"),
            decoder=self._find("decoder*.onnx"),
            joiner=self._find("joiner*.onnx"),
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
            decoding_method="greedy_search",
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Feed audio in chunks to the streaming recogniser; return final text."""
        if audio.size == 0:
            return ""

        recognizer = self._recognizer
        stream = recognizer.create_stream()

        # Feed all audio in fixed-size chunks to simulate streaming
        offset = 0
        while offset < len(audio):
            chunk = audio[offset : offset + self.chunk_frames]
            stream.accept_waveform(sample_rate, chunk)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            offset += self.chunk_frames

        # Signal end-of-stream and flush remaining frames
        tail_paddings = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        return result.text.strip()
