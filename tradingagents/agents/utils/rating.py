"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re
from typing import Tuple


# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)

# Matches a rating word in a structured recommendation context:
# - Bold markdown: **Buy**, **Hold**, etc.
# - After recommendation keywords: "recommend: Buy", "stance: Hold", "decision: Sell"
# - Standalone bullet/heading: "- Hold" or "## Hold"
_RATING_CONTEXT_RE = re.compile(
    r"(?:\*\*(\w+)\*\*"
    r"|(?:recommend(?:ation)?|stance|decision|conclusion|action)\s*[:\-]\s*\**(\w+)\**"
    r"|^[-#\s]*(\w+)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Three-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Look for a rating word in a structured context (bold, after recommendation
       keywords, or as a standalone heading/bullet) — avoids picking up rating
       words mentioned in passing prose (e.g. "an outright buy or sell").
    3. Fall back to default.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for m in _RATING_CONTEXT_RE.finditer(text):
        word = next((g for g in m.groups() if g), None)
        if word and word.lower() in _RATING_SET:
            return word.capitalize()

    return default
