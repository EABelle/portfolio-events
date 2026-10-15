"""Fetch, extract, store — incrementally.

Ties the three pieces together for a list of tickers and a date range, both supplied
by the caller. Two properties matter more than the plumbing:

- **Incremental.** Sources already in the store are skipped before anything is
  fetched, so re-running costs nothing and nobody's rate limit is spent twice.
- **`published_at` comes from the document.** Never from the moment of ingestion.
  `ingested_at` records that separately, which is what lets a later query ask what was
  actually known at a past date.

One document failing never stops the run: filings are missing, hosts time out, and a
partial ingestion is worth more than none. Every failure is reported back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .documents import Document, find_filings, find_fomc_statements, to_text
from .extract import extract
from .http import FetchError, PoliteClient
from .store import EventStore


@dataclass
class Report:
    """What an ingestion run did, in enough detail to act on."""
    found: int = 0
    skipped: int = 0            # already stored
    ingested: int = 0           # documents that produced at least one event
    events: int = 0
    discarded: int = 0          # events whose quote did not check out
    unknown_tickers: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def ingest(store: EventStore, client: PoliteClient, tickers: list[str],
           start: date, end: date, include_fomc: bool = True,
           now: datetime | None = None, _extract=extract) -> Report:
    """Ingests 8-Ks for `tickers` (and FOMC statements) published in the range."""
    report = Report()
    documents: list[Document] = []

    try:
        filings, unknown = find_filings(client, tickers, start, end)
        documents += filings
        report.unknown_tickers = unknown
    except (FetchError, ValueError) as e:
        report.failures.append(("EDGAR index", f"{type(e).__name__}: {e}"))

    if include_fomc:
        try:
            documents += find_fomc_statements(client, start, end)
        except (FetchError, ValueError) as e:
            report.failures.append(("FOMC feed", f"{type(e).__name__}: {e}"))

    report.found = len(documents)
    already = store.known_sources()
    stamp = now or datetime.now(timezone.utc)

    for document in documents:
        if document.source_id in already:
            report.skipped += 1
            continue
        try:
            text = to_text(client.get_text(document.url))
        except FetchError as e:
            report.failures.append((document.source_id, str(e)))
            continue
        if not text.strip():
            report.failures.append((document.source_id, "no readable text in the document"))
            continue

        result = _extract(text, document.source_id, [document.ticker])
        report.discarded += len(result.discarded)
        if not result.ok:
            report.failures.append((document.source_id, result.message or result.status))
            continue
        if not result.events:
            continue
        written = store.add(
            result.events, document.source_id, document.published_at,
            source_url=document.url, ingested_at=stamp,
        )
        report.events += written
        report.ingested += 1
    return report
