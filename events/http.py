"""A deliberately polite HTTP client.

The SEC publishes access rules for EDGAR and enforces them: requests must carry a
User-Agent identifying who you are and how to reach you, and the ceiling is 10
requests per second. Ignoring either gets you blocked, and being blocked is not a
transient error — it is a reputation problem for whoever owns the IP.

So the rules are structural here rather than advisory:

- a contact User-Agent is **required** to construct the client, and one that does not
  look like a contact is refused;
- the rate limit is enforced by the client, defaulting to half the published ceiling;
- only known document hosts can be reached, so a URL that arrives from fetched content
  cannot redirect this anywhere interesting;
- responses are size-capped, because a filing index can point at something enormous.

Standard library only: this repo's runtime dependencies are the LLM SDK and pydantic,
and fetching does not need to add to them.
"""
from __future__ import annotations

import gzip
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit

# Document sources with a reliable publication date. Nothing else may be fetched.
ALLOWED_HOSTS = frozenset({"www.sec.gov", "data.sec.gov", "www.federalreserve.gov"})
SEC_CEILING_PER_SECOND = 10.0      # published by the SEC
DEFAULT_RATE = 5.0                 # half of it, on purpose
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 12 * 1024 * 1024
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """The document could not be fetched. Never raised for policy problems."""


class PolicyError(ValueError):
    """The request would break the source's access rules, so it is not made."""


def check_user_agent(user_agent: str) -> str:
    """The SEC asks for 'Company Name contact@domain.com'. A bare library name is
    what gets an IP blocked, so refuse it rather than discover that in production."""
    ua = (user_agent or "").strip()
    if "@" not in ua or " " not in ua or len(ua) < 10:
        raise PolicyError(
            "user_agent must identify you and how to reach you, e.g. "
            "'Jane Doe jane@example.com'; EDGAR blocks anonymous clients"
        )
    return ua


def check_url(url: str) -> str:
    parts = urlsplit(str(url))
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise PolicyError(f"refusing to fetch {url!r}: not an allowed https source")
    if parts.username or parts.password:
        raise PolicyError("refusing to fetch a URL carrying credentials")
    return url


@dataclass
class PoliteClient:
    user_agent: str
    rate_per_second: float = DEFAULT_RATE
    timeout: float = DEFAULT_TIMEOUT
    max_bytes: int = DEFAULT_MAX_BYTES
    attempts: int = 3
    _last_request: float = field(default=0.0, repr=False)
    _sleep = staticmethod(time.sleep)
    _now = staticmethod(time.monotonic)

    def __post_init__(self) -> None:
        self.user_agent = check_user_agent(self.user_agent)
        if not 0 < self.rate_per_second <= SEC_CEILING_PER_SECOND:
            raise PolicyError(
                f"rate_per_second must be in (0, {SEC_CEILING_PER_SECOND}]; "
                f"the SEC publishes that ceiling and enforces it"
            )

    def _wait_turn(self) -> None:
        gap = 1.0 / self.rate_per_second
        waited = self._now() - self._last_request
        if waited < gap:
            self._sleep(gap - waited)
        self._last_request = self._now()

    def get(self, url: str) -> bytes:
        """Fetches a document, obeying the rate limit. Raises FetchError on failure."""
        check_url(url)
        request = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": urlsplit(url).hostname,
        })
        last = None
        for attempt in range(self.attempts):
            self._wait_turn()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read(self.max_bytes + 1)
                    if len(body) > self.max_bytes:
                        raise FetchError(f"{url} is larger than {self.max_bytes} bytes")
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in RETRY_STATUSES:
                    raise FetchError(f"{url} returned HTTP {e.code}") from e
            except FetchError:
                raise
            except Exception as e:  # noqa: BLE001  network, DNS, timeout
                last = e
            if attempt + 1 < self.attempts:
                self._sleep(2.0 * (attempt + 1))
        raise FetchError(f"{url} failed after {self.attempts} attempts: {type(last).__name__}")

    def get_text(self, url: str) -> str:
        return self.get(url).decode("utf-8", errors="replace")
