"""Turning one document into verified events.

The guarantee this module exists to provide: **every stored event can be traced back
to a fragment that really appears in its source.** The model is asked to quote
verbatim, and the quote is then looked for in the document. Anything that cannot be
found is dropped.

That check is cheap and it is the difference between a summary you can audit and one
you have to trust. It does not make the extraction correct — a real quote can still be
summarised badly — but it makes invention visible.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .llm import generate
from .schema import KINDS, Event, Extraction

INSTRUCTIONS = (
    "Extract the discrete events this document reports. One event per distinct fact. "
    "For each one: the ticker it concerns, a kind from the allowed list, one factual "
    "sentence summarising it, and a quote copied VERBATIM from the document that "
    "supports the summary. The quote must be an exact substring of the text you were "
    "given, not a paraphrase; it will be checked, and any event whose quote cannot be "
    "found is discarded. Do not interpret whether an event is good or bad news, and do "
    "not estimate its effect on the price. If the document reports nothing of "
    "substance, return no events."
)
# Quotes shorter than this match too easily by accident and prove nothing.
MIN_QUOTE = 25


@dataclass
class Result:
    events: list[Event] = field(default_factory=list)
    discarded: list[tuple[Event, str]] = field(default_factory=list)
    status: str = "ok"
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def normalise(text: str) -> str:
    """Collapses the differences that make an honest quote fail a literal match:
    smart quotes, non-breaking spaces, line wrapping, casing."""
    text = unicodedata.normalize("NFKC", text)
    for fancy, plain in (("“", '"'), ("”", '"'), ("‘", "'"),
                         ("’", "'"), ("–", "-"), ("—", "-")):
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().casefold()


def verify(event: Event, document: str, tickers: set[str] | None = None) -> str | None:
    """Returns the reason to discard the event, or None if it holds up."""
    quote = normalise(event.quote)
    if len(quote) < MIN_QUOTE:
        return f"quote too short to prove anything ({len(quote)} chars)"
    if quote not in normalise(document):
        return "quote does not appear in the document"
    if event.kind not in KINDS:
        return f"unknown kind {event.kind!r}"
    if tickers is not None and event.ticker not in tickers:
        # The caller asked about specific names; anything else is the model wandering.
        return f"ticker {event.ticker!r} was not among the ones asked about"
    if not event.summary.strip():
        return "empty summary"
    return None


def extract(document: str, source_id: str, tickers: list[str] | None = None,
            model: str | None = None, _generate=generate) -> Result:
    """Extracts events from `document`, keeping only those whose quote checks out.

    `tickers` is what the caller is asking about. This library never decides that:
    the list comes from outside, which is what keeps it free of portfolio knowledge.
    """
    if not document or not document.strip():
        return Result(status="empty", message="the document is empty")

    evidence = {
        "sources": {source_id: {"text": document}},
        "tickers_asked_about": tickers or [],
        "allowed_kinds": sorted(KINDS),
    }
    kwargs = {"model": model} if model else {}
    answer = _generate(evidence, Extraction, INSTRUCTIONS, **kwargs)
    if answer["status"] != "ok":
        return Result(status=answer["status"], message=answer.get("message"))

    wanted = set(tickers) if tickers else None
    result = Result()
    for event in answer["parsed"].events:
        if reason := verify(event, document, wanted):
            result.discarded.append((event, reason))
        else:
            result.events.append(event)
    return result
