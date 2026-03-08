"""
Sherpa-ONNX backend.

Supports two model types:
  - "online"  (default): streaming Zipformer transducer via OnlineRecognizer.
    Simulates streaming by feeding fixed-size chunks; matches live-asr-sherpa.
  - "nemo_transducer": offline NeMo Parakeet TDT via OfflineRecognizer.
    Passes full audio at once; use with --sherpa-model-type nemo_transducer.
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
        chunk_size: float = 0.1,  # seconds per simulated chunk (online only)
        model_type: str = "online",  # "online" or "nemo_transducer"
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_size)
        self.model_type = model_type
        self._recognizer = None

    def _find(self, pattern: str) -> str:
        matches = sorted(self.model_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No file matching '{pattern}' in {self.model_dir}"
            )
        return str(matches[0])

    def _is_nemo_transducer(self) -> bool:
        """Return True if the model dir looks like an offline NeMo transducer.

        Online zipformer encoders always have '-chunk-' in the filename
        (e.g. encoder-epoch-99-avg-1-chunk-16-left-128.onnx).  NeMo parakeet
        encoders use a simple name like encoder.onnx / encoder.int8.onnx.
        """
        matches = sorted(self.model_dir.glob("encoder*.onnx"))
        return bool(matches) and not any("-chunk-" in m.name for m in matches)

    def load(self) -> None:
        import sherpa_onnx

        # Auto-detect NeMo transducer when model_type was left at the default
        # "online" but the encoder file doesn't carry the zipformer chunk suffix.
        effective_type = self.model_type
        if effective_type == "online" and self._is_nemo_transducer():
            effective_type = "nemo_transducer"

        if effective_type == "nemo_transducer":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=self._find("tokens.txt"),
                encoder=self._find("encoder*.onnx"),
                decoder=self._find("decoder*.onnx"),
                joiner=self._find("joiner*.onnx"),
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
            )
            self.model_type = "nemo_transducer"  # keep transcribe() in sync
        else:
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
        if audio.size == 0:
            return ""

        if self.model_type == "nemo_transducer":
            return self._transcribe_offline(audio, sample_rate)
        return self._transcribe_online(audio, sample_rate)

    def _transcribe_offline(self, audio: np.ndarray, sample_rate: int) -> str:
        """Pass full audio to offline NeMo recogniser."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def _transcribe_online(self, audio: np.ndarray, sample_rate: int) -> str:
        """Feed audio in chunks to the streaming recogniser; return final text."""
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
        # sherpa-onnx >= 1.10 returns a plain string from get_result()
        if isinstance(result, str):
            return result.strip()
        return result.text.strip()
