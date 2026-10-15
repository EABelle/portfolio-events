"""Turns filings and statements into verified, dated events.

Portfolio-agnostic on purpose: the caller passes the tickers. Nothing here knows
what anyone holds.
"""
from .extract import extract
from .schema import Event, StoredEvent
from .store import EventStore

__all__ = ["extract", "Event", "StoredEvent", "EventStore"]
