"""Offline tests for fetching and ingestion: python -m pytest -q

No network. The client is stubbed, so what is tested is the policy around fetching —
the part that gets an IP blocked when it is wrong.
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from events.documents import (
    FOMC_FEED,
    SUBMISSIONS_URL,
    TICKERS_URL,
    cik_by_ticker,
    find_filings,
    find_fomc_statements,
    to_text,
)
from events.extract import Result
from events.http import (
    ALLOWED_HOSTS,
    SEC_CEILING_PER_SECOND,
    FetchError,
    PolicyError,
    PoliteClient,
    check_url,
    check_user_agent,
)
from events.ingest import ingest
from events.schema import Event
from events.store import EventStore

CONTACT = "Jane Doe jane@example.com"

TICKERS_JSON = json.dumps({
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 21344, "ticker": "KO", "title": "Coca-Cola Co"},
})
SUBMISSIONS_JSON = json.dumps({
    "filings": {"recent": {
        "accessionNumber": ["0000320193-26-000100", "0000320193-26-000090", "0000320193-26-000050"],
        "filingDate": ["2026-10-14", "2026-10-02", "2026-08-01"],
        "form": ["8-K", "10-Q", "8-K"],
        "primaryDocument": ["aapl-8k.htm", "aapl-10q.htm", "aapl-old.htm"],
    }}
})
FOMC_XML = """<?xml version="1.0"?><rss><channel>
<item><title>Federal Open Market Committee statement</title>
      <link>https://www.federalreserve.gov/newsevents/pressreleases/monetary20261014a.htm</link>
      <pubDate>Wed, 14 Oct 2026 18:00:00 GMT</pubDate></item>
<item><title>Speech by Governor Smith on community banking</title>
      <link>https://www.federalreserve.gov/newsevents/speech/smith20261013a.htm</link>
      <pubDate>Tue, 13 Oct 2026 14:00:00 GMT</pubDate></item>
</channel></rss>"""
FILING_HTML = """<html><head><title>8-K</title><style>.x{color:red}</style></head>
<body><script>var a=1;</script>
<p>Item 2.02 Results of Operations.</p>
<p>On October 14, 2026, Apple Inc. announced revenue of $4.1&nbsp;billion for the
quarter, an increase of 12 percent.</p></body></html>"""


class FakeClient:
    """Stands in for PoliteClient without touching the network."""

    def __init__(self, pages=None, fail=()):
        self.pages = pages or {}
        self.fail = set(fail)
        self.requested: list[str] = []

    def get_text(self, url: str) -> str:
        self.requested.append(url)
        if url in self.fail:
            raise FetchError(f"{url} failed")
        if url in self.pages:
            return self.pages[url]
        return FILING_HTML


def _pages():
    return {TICKERS_URL: TICKERS_JSON,
            SUBMISSIONS_URL.format(cik=320193): SUBMISSIONS_JSON,
            FOMC_FEED: FOMC_XML}


# --- The access policy is structural, not advisory -------------------------
@pytest.mark.parametrize("ua", ["", "portfolio-events", "bot/1.0", "events@", "a@b"])
def test_a_user_agent_without_a_contact_is_refused(ua):
    """EDGAR blocks anonymous clients; refuse before that happens, not after."""
    with pytest.raises(PolicyError, match="user_agent"):
        check_user_agent(ua)
    with pytest.raises(PolicyError):
        PoliteClient(user_agent=ua)


def test_a_proper_contact_is_accepted():
    assert check_user_agent(CONTACT) == CONTACT
    assert PoliteClient(user_agent=CONTACT).user_agent == CONTACT


def test_the_published_rate_ceiling_cannot_be_exceeded():
    with pytest.raises(PolicyError, match="ceiling"):
        PoliteClient(user_agent=CONTACT, rate_per_second=SEC_CEILING_PER_SECOND + 1)
    with pytest.raises(PolicyError):
        PoliteClient(user_agent=CONTACT, rate_per_second=0)
    # the default sits deliberately below it
    assert PoliteClient(user_agent=CONTACT).rate_per_second < SEC_CEILING_PER_SECOND


def test_the_rate_limit_actually_waits():
    client = PoliteClient(user_agent=CONTACT, rate_per_second=2.0)
    slept, clock = [], [0.0]
    client._sleep = slept.append
    client._now = lambda: clock[0]
    client._wait_turn()
    client._wait_turn()          # immediately after: must wait half a second
    assert slept and slept[0] == pytest.approx(0.5)


@pytest.mark.parametrize("url", [
    "http://www.sec.gov/x",                        # no TLS
    "https://sec.gov.evil.example/x",              # lookalike host
    "https://raw.githubusercontent.com/x",         # not a document source
    "https://user:pass@www.sec.gov/x",             # credentials in the URL
    "file:///etc/passwd",
])
def test_only_known_document_sources_can_be_fetched(url):
    with pytest.raises(PolicyError):
        check_url(url)


def test_the_allowed_hosts_are_the_two_sources_with_publication_dates():
    assert ALLOWED_HOSTS == {"www.sec.gov", "data.sec.gov", "www.federalreserve.gov"}
    for url in (TICKERS_URL, FOMC_FEED, SUBMISSIONS_URL.format(cik=1)):
        assert check_url(url) == url


# --- Turning filing HTML into quotable text --------------------------------
def test_script_and_style_never_reach_the_text():
    text = to_text(FILING_HTML)
    assert "var a=1" not in text and "color:red" not in text
    assert "Results of Operations" in text


def test_a_quote_spanning_a_line_wrap_still_verifies():
    """Filing HTML hard-wraps sentences. The text keeps those wraps, so what matters
    is that verification sees through them — it normalises whitespace before matching."""
    from events.extract import verify

    text = to_text(FILING_HTML)
    assert "revenue of $4.1 billion for the\nquarter" in text      # the wrap survives
    assert verify(_event(), text) is None                          # the quote still holds
    assert "\xa0" not in text and "  " not in text                 # nbsp and runs gone


def test_malformed_markup_still_yields_what_it_can():
    assert "hello" in to_text("<p>hello<div><span>world")


# --- EDGAR -----------------------------------------------------------------
def test_tickers_map_to_cik():
    assert cik_by_ticker(FakeClient(_pages())) == {"AAPL": 320193, "KO": 21344}


def test_a_mapping_in_an_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="empty or in an unknown shape"):
        cik_by_ticker(FakeClient({TICKERS_URL: "[]"}))


def test_only_the_requested_forms_and_dates_come_back():
    client = FakeClient(_pages())
    docs, unknown = find_filings(client, ["AAPL"], date(2026, 10, 1), date(2026, 10, 31))
    assert unknown == []
    assert len(docs) == 1                       # the 10-Q and the August 8-K are out
    doc = docs[0]
    assert doc.form == "8-K"
    assert doc.published_at == date(2026, 10, 14)
    assert doc.ticker == "AAPL"
    assert doc.source_id == "0000320193-26-000100"
    assert doc.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000100/aapl-8k.htm"
    )


def test_a_ticker_the_sec_does_not_know_is_reported_not_guessed():
    docs, unknown = find_filings(FakeClient(_pages()), ["AAPL", "NOTREAL"],
                                 date(2026, 10, 1), date(2026, 10, 31))
    assert unknown == ["NOTREAL"]
    assert [d.ticker for d in docs] == ["AAPL"]


def test_the_publication_date_comes_from_the_filing_not_from_now():
    docs, _ = find_filings(FakeClient(_pages()), ["AAPL"], date(2026, 1, 1), date(2026, 12, 31))
    assert {d.published_at for d in docs} == {date(2026, 10, 14), date(2026, 8, 1)}


# --- FOMC ------------------------------------------------------------------
def test_only_policy_statements_come_from_the_feed():
    docs = find_fomc_statements(FakeClient(_pages()), date(2026, 10, 1), date(2026, 10, 31))
    assert len(docs) == 1                       # the speech is not a statement
    assert docs[0].ticker == "FOMC"
    assert docs[0].published_at == date(2026, 10, 14)
    assert docs[0].form == "FOMC statement"


def test_statements_outside_the_range_are_left_out():
    assert find_fomc_statements(FakeClient(_pages()), date(2026, 1, 1), date(2026, 6, 30)) == []


# --- Ingestion -------------------------------------------------------------
def _event(ticker="AAPL"):
    return Event(ticker=ticker, kind="earnings",
                 summary="Apple reported revenue of $4.1 billion, up 12 percent.",
                 quote="announced revenue of $4.1 billion for the quarter")


def _extract_ok(text, source_id, tickers=None, **kw):
    return Result(events=[_event(tickers[0] if tickers else "AAPL")])


@pytest.fixture
def store():
    with EventStore() as s:
        yield s


def test_a_run_fetches_extracts_and_stores(store):
    client = FakeClient(_pages())
    report = ingest(store, client, ["AAPL"], date(2026, 10, 1), date(2026, 10, 31),
                    now=datetime(2026, 10, 15, tzinfo=UTC), _extract=_extract_ok)
    assert report.ok
    assert report.found == 2            # one 8-K and one FOMC statement
    assert report.ingested == 2 and report.events == 2
    assert store.count() == 2
    assert {e.ticker for e in store.find()} == {"AAPL", "FOMC"}


def test_rerunning_skips_what_is_already_stored(store):
    client = FakeClient(_pages())
    args = (store, client, ["AAPL"], date(2026, 10, 1), date(2026, 10, 31))
    ingest(*args, now=datetime(2026, 10, 15, tzinfo=UTC), _extract=_extract_ok)
    before = len(client.requested)

    again = ingest(*args, now=datetime(2026, 10, 16, tzinfo=UTC), _extract=_extract_ok)
    assert again.skipped == 2 and again.ingested == 0
    assert store.count() == 2
    # the documents themselves are not fetched a second time
    assert len(client.requested) - before == 3   # tickers, submissions and the feed only


def test_published_and_ingested_dates_are_recorded_separately(store):
    ingest(store, FakeClient(_pages()), ["AAPL"], date(2026, 10, 1), date(2026, 10, 31),
           now=datetime(2026, 11, 20, 9, tzinfo=UTC), _extract=_extract_ok)
    stored = store.find(["AAPL"])[0]
    assert stored.published_at == date(2026, 10, 14)       # from the filing
    assert stored.ingested_at.date() == date(2026, 11, 20)  # from the run
    assert stored.is_backfilled is True


def test_one_broken_document_does_not_stop_the_run(store):
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000100/aapl-8k.htm"
    client = FakeClient(_pages(), fail=[url])
    report = ingest(store, client, ["AAPL"], date(2026, 10, 1), date(2026, 10, 31),
                    _extract=_extract_ok)
    assert not report.ok
    assert report.failures[0][0] == "0000320193-26-000100"
    assert report.ingested == 1                 # the FOMC statement still went in
    assert store.count() == 1


def test_a_failing_index_is_reported_and_the_other_source_still_runs(store):
    client = FakeClient(_pages(), fail=[TICKERS_URL])
    report = ingest(store, client, ["AAPL"], date(2026, 10, 1), date(2026, 10, 31),
                    _extract=_extract_ok)
    assert any(source == "EDGAR index" for source, _ in report.failures)
    assert report.ingested == 1                 # FOMC is independent


def test_events_whose_quote_failed_are_counted_not_stored(store):
    def extract_all_discarded(text, source_id, tickers=None, **kw):
        return Result(events=[], discarded=[(_event(), "quote does not appear in the document")])

    report = ingest(store, FakeClient(_pages()), ["AAPL"], date(2026, 10, 1),
                    date(2026, 10, 31), _extract=extract_all_discarded)
    assert report.discarded == 2 and report.events == 0
    assert store.count() == 0


def test_an_unavailable_model_is_a_failure_not_silent_success(store):
    def extract_unavailable(text, source_id, tickers=None, **kw):
        return Result(status="unavailable", message="timeout")

    report = ingest(store, FakeClient(_pages()), ["AAPL"], date(2026, 10, 1),
                    date(2026, 10, 31), _extract=extract_unavailable)
    assert not report.ok
    assert all(reason == "timeout" for _, reason in report.failures)
    assert store.count() == 0


def test_fomc_can_be_left_out(store):
    report = ingest(store, FakeClient(_pages()), ["AAPL"], date(2026, 10, 1),
                    date(2026, 10, 31), include_fomc=False, _extract=_extract_ok)
    assert report.found == 1
    assert store.find(["FOMC"]) == []


def test_the_caller_still_supplies_the_tickers(store):
    """Nothing in this path infers which names matter."""
    report = ingest(store, FakeClient(_pages()), [], date(2026, 10, 1), date(2026, 10, 31),
                    include_fomc=False, _extract=_extract_ok)
    assert report.found == 0 and store.count() == 0


def test_a_filing_without_a_named_primary_document_falls_back_to_the_full_text():
    """primaryDocument is optional in the SEC index; a filing must not be lost over it."""
    without = json.dumps({"filings": {"recent": {
        "accessionNumber": ["0000320193-26-000100"],
        "filingDate": ["2026-10-14"],
        "form": ["8-K"],
    }}})
    pages = _pages() | {SUBMISSIONS_URL.format(cik=320193): without}
    docs, _ = find_filings(FakeClient(pages), ["AAPL"], date(2026, 10, 1), date(2026, 10, 31))
    assert len(docs) == 1
    assert docs[0].url.endswith("/0000320193-26-000100.txt")
    assert docs[0].source_id == "0000320193-26-000100"


def test_the_three_fields_the_index_must_have_are_required():
    broken = json.dumps({"filings": {"recent": {"accessionNumber": ["x"]}}})
    pages = _pages() | {SUBMISSIONS_URL.format(cik=320193): broken}
    with pytest.raises(ValueError, match="unexpected submissions shape"):
        find_filings(FakeClient(pages), ["AAPL"], date(2026, 10, 1), date(2026, 10, 31))
