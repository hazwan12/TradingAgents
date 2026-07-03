"""Google News RSS fetcher for ticker-specific news headlines.

Uses Google News RSS search (no API key, no auth, no registration required).
Covers a broad set of sources: Bloomberg, Reuters, CNBC, WSJ, FT, etc., giving
a wider institutional news picture than Yahoo Finance alone.

Two search modes:
  - General: recent news mentioning the ticker + "stock"
  - Seeking Alpha: articles published on seekingalpha.com about the ticker,
    fetched via Google's aggregation to avoid SA's direct RSS restrictions.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "": "",
}


def _fetch_rss(url: str, timeout: float = 15.0) -> Optional[ET.Element]:
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/rss+xml, application/xml, text/xml"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return ET.fromstring(resp.read())
    except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.warning("Google News RSS fetch failed (%s): %s", url, exc)
        return None


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", unescape(text or ""))
    return " ".join(text.split()).strip()


def _parse_items(root: ET.Element, max_items: int) -> list[dict]:
    items = []
    channel = root.find("channel")
    if channel is None:
        return items
    for item in channel.findall("item")[:max_items]:
        title = _clean(item.findtext("title", ""))
        pub_date = item.findtext("pubDate", "")
        source_el = item.find("source")
        source = source_el.text if source_el is not None else ""
        description = _clean(item.findtext("description", ""))
        if title:
            items.append({
                "title": title,
                "source": source,
                "date": pub_date[:16] if pub_date else "",
                "description": description[:200] + "…" if len(description) > 200 else description,
            })
    return items


def fetch_google_news(ticker: str, max_items: int = 10, timeout: float = 15.0) -> str:
    """Fetch recent Google News headlines for ``ticker`` and return a
    formatted plaintext block ready for prompt injection."""
    query = quote_plus(f"{ticker} stock")
    url = _GOOGLE_NEWS_RSS.format(query=query)
    root = _fetch_rss(url, timeout)
    if root is None:
        return f"<Google News unavailable for {ticker.upper()}>"

    items = _parse_items(root, max_items)
    if not items:
        return f"<no Google News articles found for {ticker.upper()}>"

    lines = [f"Google News — {len(items)} recent articles mentioning {ticker.upper()}:"]
    for it in items:
        src = f" · {it['source']}" if it["source"] else ""
        lines.append(f"  [{it['date']}{src}] {it['title']}")
        if it["description"]:
            lines.append(f"    {it['description']}")
    return "\n".join(lines)


_SA_RSS = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"


def fetch_seeking_alpha(ticker: str, max_items: int = 5, timeout: float = 15.0) -> str:
    """Fetch recent Seeking Alpha articles for ``ticker`` and return a
    formatted plaintext block ready for prompt injection.

    Uses SA's combined RSS endpoint directly; falls back to Google News
    site-filtered query if SA blocks the request.
    """
    url = _SA_RSS.format(ticker=ticker.upper())
    root = _fetch_rss(url, timeout)

    # Fall back to Google News site filter if SA returns HTML (bot-blocked) or fails
    if root is None or root.tag.lower() == "html":
        query = quote_plus(f'"{ticker}" seekingalpha')
        root = _fetch_rss(_GOOGLE_NEWS_RSS.format(query=query), timeout)
        if root is None:
            return f"<Seeking Alpha unavailable for {ticker.upper()}>"

    items = _parse_items(root, max_items)
    if not items:
        return f"<no Seeking Alpha articles found for {ticker.upper()}>"

    lines = [f"Seeking Alpha — {len(items)} recent articles for {ticker.upper()}:"]
    for it in items:
        lines.append(f"  [{it['date']}] {it['title']}")
        if it["description"]:
            lines.append(f"    {it['description']}")
    return "\n".join(lines)
