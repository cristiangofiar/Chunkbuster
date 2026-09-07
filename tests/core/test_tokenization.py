from __future__ import annotations

from chunkbuster import TextTokenizer


def test_word_tokenizer_supports_case_accents_stop_words_and_min_length() -> None:
    tokenizer = TextTokenizer(
        strip_accents=True,
        min_length=3,
        stop_words=frozenset({"THE"}),
    )

    assert tokenizer.tokenize("THE Café, e-mail y NIÑO") == (
        "cafe",
        "e-mail",
        "nino",
    )


def test_whitespace_tokenizer_can_preserve_case_and_punctuation() -> None:
    tokenizer = TextTokenizer(mode="whitespace", case_sensitive=True)

    assert tokenizer.prepare_query("Red, SHOES") == ("Red,", "SHOES")
    assert tokenizer.prepare_documents(("A B", "C")) == (("A", "B"), ("C",))
