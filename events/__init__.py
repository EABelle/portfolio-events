"""Turns filings and statements into verified, dated events.

Portfolio-agnostic on purpose: the caller passes the tickers. Nothing here knows
what anyone holds.
"""
from .documents import Document, find_filings, find_fomc_statements, to_text
from .extract import extract
from .http import PoliteClient
from .ingest import Report, ingest
from .schema import Event, StoredEvent
from .store import EventStore

__all__ = [
    "Document", "Event", "EventStore", "PoliteClient", "Report", "StoredEvent",
    "extract", "find_filings", "find_fomc_statements", "ingest", "to_text",
]
