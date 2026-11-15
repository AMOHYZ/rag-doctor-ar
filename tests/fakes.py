"""Offline fakes so tests never touch Ollama or the network."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rag.common import Completion

_DIM = 64


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    model = "fake"

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, sum(word.encode()) % _DIM] += 1.0
        return out


class FakeLLM:
    """Returns a canned reply and records the prompts it was given."""

    def __init__(self, reply: str = "canned answer") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, json_mode: bool = False) -> Completion:
        self.prompts.append(prompt)
        return Completion(text=self.reply, prompt_tokens=len(prompt.split()), completion_tokens=2)
