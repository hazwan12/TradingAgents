#!/usr/bin/env python3
"""Virtual portfolio state and paper trading track record for the Telegram channel broadcast.

Tracks simulated positions: opens on Buy, closes on Sell, ignores Hold.
Records entry/exit prices and per-trade returns so subscribers can see
a running track record alongside the daily holdings digest.

State file: ~/.tradingagents/broadcast_positions.json
  - One key per ticker: {status, opened_date, universes, entry_price, exit_price, ...}
  - "__stats__" key: closed-trade ledger and aggregate performance metrics
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

POSITIONS_PATH = Path.home() / ".tradingagents" / "broadcast_positions.json"

REAL_UNIVERSE_TAGS = {"full", "sp500", "nasdaq100"}


def load_positions() -> Dict[str, dict]:
    if not POSITIONS_PATH.exists():
        return {}
    try:
        return json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_positions(positions: Dict[str, dict]):
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_PATH.write_text(json.dumps(positions, indent=2), encoding="utf-8")


def get_stats(positions: Dict[str, dict]) -> dict:
    return positions.get("__stats__", {
        "closed_trades": [],
        "cumulative_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "total_trades": 0,
    })


def _update_stats(positions: Dict[str, dict], trade: dict) -> dict:
    stats = get_stats(positions)
    stats["closed_trades"].append(trade)
    trades = stats["closed_trades"]
    stats["total_trades"] = len(trades)
    returns = [t["return_pct"] for t in trades if t.get("return_pct") is not None]
    if returns:
        stats["cumulative_return_pct"] = round(sum(returns) / len(returns), 2)
        stats["win_rate_pct"] = round(100 * sum(1 for r in returns if r > 0) / len(returns), 1)
    return stats


def update_positions(
    date: str,
    results: Dict[str, dict],
    tags: Dict[str, List[str]],
    prices: Optional[Dict[str, float]] = None,
    persist: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Apply today's ratings to the virtual portfolio.

    Args:
        date: analysis date
        results: {ticker: {rating, action, executive_summary, ...}}
        tags: {ticker: [origin tags]} from market_scan
        prices: {ticker: current_price} — used to record entry/exit for paper trading
        persist: write updated state to disk (False for dry-run previews)

    Returns:
        (opened, closed, still_open) — each a list of enriched dicts
    """
    positions = load_positions()
    prices = prices or {}
    opened, closed, still_open = [], [], []

    for ticker, result in results.items():
        action = result.get("action", "Hold")
        ticker_tags = tags.get(ticker, [])
        pos = positions.get(ticker, {})
        is_open = pos.get("status") == "open"
        from_real_universe = bool(REAL_UNIVERSE_TAGS & set(ticker_tags))
        current_price = prices.get(ticker)

        entry = {
            "ticker": ticker,
            "action": action,
            "rating": result.get("rating", ""),
            "executive_summary": result.get("executive_summary", ""),
            "universes": [t for t in ticker_tags if t in REAL_UNIVERSE_TAGS],
            "current_price": current_price,
        }

        if not is_open and action == "Buy" and from_real_universe:
            positions[ticker] = {
                "status": "open",
                "opened_date": date,
                "universes": entry["universes"],
                "entry_price": current_price,
                "exit_price": None,
                "return_pct": None,
                "closed_date": None,
            }
            entry["entry_price"] = current_price
            opened.append(entry)

        elif is_open and action == "Sell":
            entry_price = pos.get("entry_price")
            return_pct = None
            if entry_price and current_price and entry_price > 0:
                return_pct = round((current_price - entry_price) / entry_price * 100, 2)

            from datetime import datetime as _dt
            opened_date = pos.get("opened_date", "")
            days_held = 0
            if opened_date:
                try:
                    days_held = (_dt.fromisoformat(date) - _dt.fromisoformat(opened_date)).days
                except ValueError:
                    pass

            positions[ticker].update({
                "status": "closed",
                "closed_date": date,
                "exit_price": current_price,
                "return_pct": return_pct,
            })

            trade_record = {
                "ticker": ticker,
                "opened": opened_date,
                "closed": date,
                "days_held": days_held,
                "entry_price": entry_price,
                "exit_price": current_price,
                "return_pct": return_pct,
            }
            positions["__stats__"] = _update_stats(positions, trade_record)

            entry.update({
                "entry_price": entry_price,
                "exit_price": current_price,
                "return_pct": return_pct,
                "days_held": days_held,
            })
            closed.append(entry)

        elif is_open:
            entry_price = pos.get("entry_price")
            return_pct = None
            if entry_price and current_price and entry_price > 0:
                return_pct = round((current_price - entry_price) / entry_price * 100, 2)
            entry.update({
                "entry_price": entry_price,
                "opened_date": pos.get("opened_date", ""),
                "return_pct": return_pct,
            })
            still_open.append(entry)

    if persist:
        save_positions(positions)

    return opened, closed, still_open
