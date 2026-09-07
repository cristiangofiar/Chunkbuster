"""Dependency-free tokenization usable by any pipeline preprocessor binding."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

Tokenization = Literal["word", "whitespace"]


@dataclass(frozen=True, slots=True)
class TextTokenizer:
    """Configurable Unicode tokenizer implementing the preprocessor contract."""

    mode: Tokenization = "word"
    case_sensitive: bool = False
    strip_accents: bool = False
    min_length: int = 1
    stop_words: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.mode not in {"word", "whitespace"}:
            raise ValueError(f"unknown tokenization mode {self.mode!r}")
        if self.min_length <= 0:
            raise ValueError("tokenizer min_length must be positive")
        object.__setattr__(
            self,
            "stop_words",
            frozenset(self._normalize(word) for word in self.stop_words),
        )

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        if self.strip_accents:
            normalized = "".join(
                character
                for character in unicodedata.normalize("NFKD", normalized)
                if not unicodedata.combining(character)
            )
        return normalized if self.case_sensitive else normalized.casefold()

    def tokenize(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise TypeError("tokenizer input must be a string")
        normalized = self._normalize(text)
        tokens = (
            re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", normalized)
            if self.mode == "word"
            else normalized.split()
        )
        return tuple(
            token
            for token in tokens
            if len(token) >= self.min_length and token not in self.stop_words
        )

    def prepare_query(self, text: str) -> tuple[str, ...]:
        return self.tokenize(text)

    def prepare_documents(self, texts) -> tuple[tuple[str, ...], ...]:
        return tuple(self.tokenize(text) for text in texts)
