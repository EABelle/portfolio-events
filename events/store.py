"""SQLite store for extracted events.

The one thing this store does that an ordinary table would not: it keeps
`published_at` and `ingested_at` apart, and lets a query ask for the world **as it was
known** on a given date.

That matters because backfilling is normal — you add a source, or re-run a month of
filings — and once a backfilled event sits next to a live one, any study built on the
table silently acquires look-ahead bias. `known_by` is the guard: it filters on
`ingested_at`, so a query about last March sees only what had actually been ingested
by last March.

No vector search here. Retrieval is by ticker and date range, which is what answering
"what happened to this name last week" needs. Semantic search can be added when a
question appears that this cannot answer.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

from .schema import Event, StoredEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    ticker       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    summary      TEXT NOT NULL,
    quote        TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    source_url   TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    UNIQUE (source_id, ticker, quote)
);
CREATE INDEX IF NOT EXISTS events_by_ticker_date ON events (ticker, published_at);
CREATE INDEX IF NOT EXISTS events_by_ingested ON events (ingested_at);
"""


class EventStore:
    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, events: list[Event], source_id: str, published_at: date,
            source_url: str = "", ingested_at: datetime | None = None) -> int:
        """Stores events. Re-ingesting the same source does not duplicate them.

        `ingested_at` defaults to now, and is the only field that may differ between
        two runs over the same document.
        """
        if not events:
            return 0
        stamp = (ingested_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        rows = [
            (e.ticker, e.kind, e.summary, e.quote, source_id, source_url,
             published_at.isoformat(), stamp)
            for e in events
        ]
        with closing(self._conn.cursor()) as cur:
            cur.executemany(
                "INSERT OR IGNORE INTO events "
                "(ticker, kind, summary, quote, source_id, source_url, published_at, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            written = cur.rowcount
        self._conn.commit()
        return written

    def find(self, tickers: list[str] | None = None, start: date | None = None,
             end: date | None = None, kinds: list[str] | None = None,
             known_by: datetime | None = None, limit: int = 500) -> list[StoredEvent]:
        """Events matching the filters, newest first.

        `known_by` restricts the answer to what had been ingested by that moment. Use
        it for anything historical; without it a backfill looks like foresight.
        """
        where, params = [], []
        if tickers:
            where.append(f"ticker IN ({','.join('?' * len(tickers))})")
            params += list(tickers)
        if start:
            where.append("published_at >= ?")
            params.append(start.isoformat())
        if end:
            where.append("published_at <= ?")
            params.append(end.isoformat())
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            params += list(kinds)
        if known_by:
            where.append("ingested_at <= ?")
            params.append(known_by.isoformat(timespec="seconds"))

        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY published_at DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with closing(self._conn.cursor()) as cur:
            rows = cur.execute(sql, params).fetchall()
        return [
            StoredEvent(
                ticker=r["ticker"], kind=r["kind"], summary=r["summary"], quote=r["quote"],
                source_id=r["source_id"], source_url=r["source_url"],
                published_at=date.fromisoformat(r["published_at"]),
                ingested_at=datetime.fromisoformat(r["ingested_at"]),
            )
            for r in rows
        ]

    def known_sources(self) -> set[str]:
        """Source ids already stored, so ingestion can skip them."""
        with closing(self._conn.cursor()) as cur:
            return {r[0] for r in cur.execute("SELECT DISTINCT source_id FROM events")}

    def count(self) -> int:
        with closing(self._conn.cursor()) as cur:
            return int(cur.execute("SELECT COUNT(*) FROM events").fetchone()[0])
