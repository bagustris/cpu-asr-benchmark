"""Abstract base class for all ASR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ASRBackend(ABC):
    """Common interface that every backend must implement."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load/initialise the model (called once before benchmarking)."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a complete audio array and return the transcript string.

        Parameters
        ----------
        audio       : float32 numpy array, shape (N,), range [-1, 1]
        sample_rate : samples per second (default 16000)

        Returns
        -------
        str : the recognised transcript
        """
