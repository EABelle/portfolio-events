"""Finding documents and turning them into plain text.

Two sources, both chosen for the same reason: they carry an official publication date.
A filing has a filing date and a statement has a release timestamp, and that is what
makes `published_at` trustworthy enough to keep apart from `ingested_at`. A source
without one cannot be used here, however interesting its content.

- **EDGAR 8-K**: material events companies are required to report.
- **FOMC**: monetary policy statements, from the Federal Reserve's press release feed.

The HTML these produce is messy — inline XBRL, nested tables, boilerplate — so the
text extraction is crude on purpose. It only has to be good enough for an LLM to quote
from, and the quote is verified against this same text, so the two stay consistent
even where the stripping is imperfect.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from xml.etree import ElementTree

from .http import PoliteClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
# Every submission has a complete text file named after its accession number. Used
# when the index does not name a primary document, so a missing optional field
# degrades to a bulkier download rather than to no filing at all.
COMPLETE_SUBMISSION = "{accession}.txt"
FOMC_FEED = "https://www.federalreserve.gov/feeds/press_monetary.xml"
FOMC_TICKER = "FOMC"
# Statements, not speeches or minutes: only the first says what policy is now.
FOMC_PATTERN = re.compile(r"federal\s+open\s+market\s+committee|fomc\s+statement", re.I)

DROP_TAGS = frozenset({"script", "style", "head", "title"})
BLOCK_TAGS = frozenset({"p", "div", "br", "tr", "td", "th", "li", "h1", "h2", "h3", "h4"})


@dataclass(frozen=True)
class Document:
    """One fetchable document with a publication date from its source."""
    source_id: str
    ticker: str
    url: str
    published_at: date
    form: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in DROP_TAGS:
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in DROP_TAGS and self._skip:
            self._skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def to_text(html: str) -> str:
    """Plain text from messy filing HTML. Whitespace is collapsed per line so that a
    quote verified against this output keeps matching it."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001  malformed markup: keep whatever was parsed
        pass
    text = "".join(parser.parts)
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line))


# --------------------------------------------------------------------------
# EDGAR
# --------------------------------------------------------------------------
def cik_by_ticker(client: PoliteClient) -> dict[str, int]:
    """Ticker -> CIK, from the SEC's own mapping file."""
    raw = json.loads(client.get_text(TICKERS_URL))
    rows = raw.values() if isinstance(raw, dict) else raw
    out: dict[str, int] = {}
    for row in rows:
        try:
            out[str(row["ticker"]).upper()] = int(row["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise ValueError("the SEC ticker mapping came back empty or in an unknown shape")
    return out


def filings_for(client: PoliteClient, ticker: str, cik: int, start: date, end: date,
                forms: tuple[str, ...] = ("8-K",)) -> list[Document]:
    """Filings of the given forms, filed between `start` and `end` inclusive.

    The date comes from the filing record, never from the moment this runs.
    """
    payload = json.loads(client.get_text(SUBMISSIONS_URL.format(cik=cik)))
    recent = (payload.get("filings") or {}).get("recent") or {}
    # Only these three are required. primaryDocument is treated as optional because
    # the submissions index is the SEC's to change, and a filing we can still reach
    # through its complete text file should not be dropped over a missing field.
    needed = ("accessionNumber", "filingDate", "form")
    if not all(k in recent for k in needed):
        raise ValueError(f"unexpected submissions shape for CIK {cik}: missing {needed}")

    accessions = recent["accessionNumber"]
    primaries = recent.get("primaryDocument") or [""] * len(accessions)
    documents = []
    for accession, filed, form, primary in zip(
        accessions, recent["filingDate"], recent["form"], primaries
    ):
        if form not in forms:
            continue
        try:
            filed_at = date.fromisoformat(str(filed))
        except ValueError:
            continue
        if not start <= filed_at <= end:
            continue
        accession = str(accession)
        documents.append(Document(
            source_id=accession,
            ticker=ticker.upper(),
            url=ARCHIVE_URL.format(
                cik=cik,
                accession=accession.replace("-", ""),
                document=str(primary) or COMPLETE_SUBMISSION.format(accession=accession),
            ),
            published_at=filed_at,
            form=str(form),
        ))
    return documents


def find_filings(client: PoliteClient, tickers: list[str], start: date, end: date,
                 forms: tuple[str, ...] = ("8-K",)) -> tuple[list[Document], list[str]]:
    """Documents for the tickers the caller asked about, plus the ones not found.

    The ticker list comes from outside. This library never decides which names matter.
    """
    mapping = cik_by_ticker(client)
    documents, unknown = [], []
    for ticker in tickers:
        cik = mapping.get(ticker.upper())
        if cik is None:
            unknown.append(ticker)
            continue
        documents += filings_for(client, ticker, cik, start, end, forms)
    return documents, unknown


# --------------------------------------------------------------------------
# FOMC
# --------------------------------------------------------------------------
def find_fomc_statements(client: PoliteClient, start: date, end: date) -> list[Document]:
    """Policy statements from the Fed's press release feed.

    The feed carries `pubDate`, which is the official release moment — the reason this
    source qualifies at all.
    """
    feed = ElementTree.fromstring(client.get_text(FOMC_FEED))
    documents = []
    for item in feed.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        raw_date = (item.findtext("pubDate") or "").strip()
        if not link or not FOMC_PATTERN.search(title):
            continue
        published = _parse_pub_date(raw_date)
        if published is None or not start <= published <= end:
            continue
        documents.append(Document(
            source_id=link.rsplit("/", 1)[-1] or link,
            ticker=FOMC_TICKER,
            url=link,
            published_at=published,
            form="FOMC statement",
        ))
    return documents


def _parse_pub_date(raw: str) -> date | None:
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None
