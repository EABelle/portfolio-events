"""The fixed event schema.

The whole point of this library is that documents become *events with a fixed shape*,
not chunks of text. A chunk can only be retrieved by resembling a question; an event
can be filtered by ticker, by kind and by date, and can be checked against its source.

Two fields carry the guarantees:

- `quote` must appear verbatim in the source document. An event whose quote cannot be
  found is discarded rather than stored, so nothing invented survives the pipeline.
- `kind` comes from a closed taxonomy. A free-text label would drift between runs and
  make filtering meaningless.

Nothing here judges anything. There is no field for whether an event is good or bad
news, on purpose: this library extracts, it does not form opinions.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Closed taxonomy. Roughly the 8-K item groups plus what the FOMC actually says.
Kind = Literal[
    "earnings",       # results, revenue, margins
    "guidance",       # outlook raised, cut or confirmed
    "mna",            # acquisition, divestment, merger
    "leadership",     # CEO, CFO, board changes
    "capital",        # dividend, buyback, share issue
    "debt",           # issuance, refinancing, covenant
    "legal",          # litigation, regulator, investigation
    "restructuring",  # layoffs, plant closures, write-downs
    "rates",          # FOMC: policy rate decision
    "balance_sheet",  # FOMC: QT, QE, reserves
    "other",
]
KINDS: frozenset[str] = frozenset(Kind.__args__)

MAX_SUMMARY = 300
MAX_QUOTE = 500


class Event(BaseModel):
    """One thing that happened, traceable to a fragment of one document."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker the event is about, or FOMC for policy")
    kind: Kind
    summary: str = Field(max_length=MAX_SUMMARY, description="One factual sentence")
    quote: str = Field(
        max_length=MAX_QUOTE,
        description="A fragment copied verbatim from the document, supporting the summary",
    )


class Extraction(BaseModel):
    """What the model returns for one document."""

    model_config = ConfigDict(extra="forbid")

    events: list[Event]
    source_ids: list[str]


class StoredEvent(BaseModel):
    """An event as it lives in the store, with both dates kept apart."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    kind: str
    summary: str
    quote: str
    source_id: str
    source_url: str
    published_at: date
    ingested_at: datetime

    @property
    def is_backfilled(self) -> bool:
        """True when the event reached the store long after it was published.

        Worth surfacing: a backfill is legitimate, but treating it as if it had been
        known on `published_at` is how look-ahead bias gets in.
        """
        return (self.ingested_at.date() - self.published_at).days > 1
