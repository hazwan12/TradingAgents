"""Reddit search fetcher for ticker-specific discussion posts.

Authentication hierarchy
------------------------
1. **OAuth2 (application-only)** — used when ``REDDIT_CLIENT_ID`` and
   ``REDDIT_CLIENT_SECRET`` are set in the environment.  Requests go to
   ``oauth.reddit.com``, bypass Reddit's WAF entirely, and carry full data
   (score, comment count, post body).  The bearer token is cached in memory
   for 55 minutes and refreshed automatically.

2. **RSS fallback** — used when OAuth credentials are absent *or* when the
   OAuth call itself fails.  The public Atom/RSS search feed is less
   aggressively gated than the JSON endpoint.  It lacks score / comment
   counts, so RSS-sourced posts are tagged and the formatter omits those
   metrics rather than printing fake zeros.

Reddit's public JSON endpoint (``www.reddit.com/r/{sub}/search.json``) is
*not* attempted without credentials — Reddit's WAF returns ``HTTP 403
Blocked`` for all unauthenticated script requests routed through the SIN CDN
edge, confirmed by inspecting the response headers (``server-timing:
reddit-ct;desc="dn=FT,p=SIN"``).

Setup (optional — RSS works without this)
------------------------------------------
1. Go to https://www.reddit.com/prefs/apps and create a new app:
   - Type: **script**
   - Name / description: anything
   - Redirect URI: http://localhost (unused for script apps)
2. Copy the client_id (shown under the app name) and client_secret.
3. Add to ``.env``::

       REDDIT_CLIENT_ID=your_client_id
       REDDIT_CLIENT_SECRET=your_client_secret

No Reddit account or user OAuth flow is needed — application-only credentials
are sufficient for read access.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_OAUTH_API = "https://oauth.reddit.com/r/{sub}/search.json?{qs}"
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Token cache — shared across threads, protected by _token_lock.
_token_cache: dict = {"token": None, "expires_at": 0.0}
_token_lock = threading.Lock()

DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

# RSS rate limiter — shared across threads. Without OAuth, the public RSS feed
# is gated process-wide regardless of which ticker/subreddit is being fetched,
# so WatchlistScanner running many tickers concurrently (one thread per
# ticker) can trip 429s even though each thread paces its own requests —
# every thread races the same Reddit-side limit independently. This lock
# enforces one RSS request at a time, at least _RSS_MIN_INTERVAL apart,
# process-wide.
_rss_lock = threading.Lock()
_rss_last_request_at = 0.0
_RSS_MIN_INTERVAL = 1.5  # seconds between any two RSS requests, across all threads
_RSS_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# OAuth2 token management
# ---------------------------------------------------------------------------


def _oauth_credentials() -> tuple[str, str] | None:
    """Return (client_id, client_secret) from env, or None if not configured."""
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    return (cid, secret) if cid and secret else None


def _get_bearer_token(timeout: float = 10.0) -> str | None:
    """Return a valid OAuth2 bearer token, fetching a new one only when needed.

    Returns None if credentials are not configured or if the token request
    fails (caller should fall back to RSS).
    """
    creds = _oauth_credentials()
    if not creds:
        return None

    with _token_lock:
        # Reuse cached token if it has more than 60 s left
        if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
            return _token_cache["token"]

        client_id, client_secret = creds
        auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        body = urlencode({"grant_type": "client_credentials"}).encode()
        req = Request(
            _TOKEN_URL,
            data=body,
            headers={
                "Authorization": f"Basic {auth}",
                "User-Agent": _UA,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            token = payload.get("access_token")
            expires_in = int(payload.get("expires_in", 3600))
            if not token:
                logger.warning("Reddit OAuth2: token response missing access_token: %s", payload)
                return None
            _token_cache["token"] = token
            _token_cache["expires_at"] = time.time() + expires_in
            logger.debug("Reddit OAuth2: new bearer token obtained (expires in %ds)", expires_in)
            return token
        except Exception as exc:
            logger.warning("Reddit OAuth2: failed to obtain bearer token: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Search query string
# ---------------------------------------------------------------------------


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",
        "limit": limit,
    })


# ---------------------------------------------------------------------------
# Timestamp / HTML helpers
# ---------------------------------------------------------------------------


def _iso_to_timestamp(iso_str: Optional[str]) -> Optional[float]:
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    if not content:
        return ""
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


# ---------------------------------------------------------------------------
# Fetch paths
# ---------------------------------------------------------------------------


def _fetch_subreddit_oauth(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    token: str,
) -> list[dict]:
    """Fetch via OAuth2 — returns full post data including scores."""
    url = _OAUTH_API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": _UA,
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Reddit OAuth fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []


def _throttle_rss():
    """Block until at least _RSS_MIN_INTERVAL has passed since the last RSS
    request from any thread."""
    global _rss_last_request_at
    with _rss_lock:
        wait = _RSS_MIN_INTERVAL - (time.time() - _rss_last_request_at)
        if wait > 0:
            time.sleep(wait)
        _rss_last_request_at = time.time()


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fallback: public Atom/RSS feed — no scores or comment counts."""
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})

    for attempt in range(_RSS_MAX_RETRIES + 1):
        _throttle_rss()
        try:
            with urlopen(req, timeout=timeout) as resp:
                root = ET.fromstring(resp.read())
            break
        except HTTPError as exc:
            if exc.code == 429 and attempt < _RSS_MAX_RETRIES:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                backoff = float(retry_after) if retry_after else 2.0 * (attempt + 1)
                logger.debug(
                    "Reddit RSS 429 for r/%s · %s — retrying in %.1fs (attempt %d/%d)",
                    sub, ticker, backoff, attempt + 1, _RSS_MAX_RETRIES,
                )
                time.sleep(backoff)
                continue
            logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
            return []
        except (URLError, TimeoutError, ET.ParseError) as exc:
            logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
            return []
    else:
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Try OAuth2 first; fall back to RSS."""
    token = _get_bearer_token(timeout=timeout)
    if token:
        posts = _fetch_subreddit_oauth(ticker, sub, limit, timeout, token)
        if posts:
            return posts
        # OAuth succeeded but returned no posts — genuine empty result, skip RSS
        logger.debug("Reddit OAuth: no posts for r/%s · %s", sub, ticker)
        return []
    # No credentials configured — go straight to RSS
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance subreddits.

    Returns a formatted plaintext block ready for prompt injection.
    Degrades gracefully — returns a placeholder string rather than raising.
    """
    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
