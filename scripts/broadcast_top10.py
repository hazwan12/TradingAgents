#!/usr/bin/env python3
"""Daily broadcast — virtual portfolio of BUY/SELL events, sent to a shared Telegram group.

Unlike daily.py/nightly_analysis.py (personalized to your holdings/budget), this is
generic: same content for every member of the group. It tracks its own simulated
positions (scripts/broadcast_positions.py) — a ticker opens on a Buy rating and closes
on a Sell rating — so the channel announces events instead of restating today's rating
every day. Shares the underlying scan with the personal pipeline via
scripts/market_scan.py's per-date cache, so running both on the same day doesn't double
the LLM cost for overlapping tickers.

Usage:
    python scripts/broadcast_top10.py                          # all 3 universes, today
    python scripts/broadcast_top10.py --date 2026-06-27
    python scripts/broadcast_top10.py --universes sp500
    python scripts/broadcast_top10.py --top 10
    python scripts/broadcast_top10.py --dry-run                # show candidates/opens/closes, skip LLM + Telegram

Schedule with cron (run after the pre-screener/nightly jobs, e.g. 4:10 PM ET):
    10 16 * * 1-5  cd /path/to/TradingAgents && python scripts/broadcast_top10.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nightly_analysis import _load_profile, _build_config
from scripts.market_scan import run_unified_scan
from scripts.broadcast_positions import load_positions, update_positions
from scripts.notifier import send_telegram_broadcast, send_telegram_photo, format_position_update_message, format_holdings_digest
from scripts.chart import generate_buy_chart
from scripts.broadcast_positions import get_stats
from tradingagents.portfolio import get_current_prices

console = Console()


def run(date: str, universes: list[str], top_n: int, dry_run: bool = False):
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]TradingAgents — Channel Broadcast[/bold cyan]\n"
            f"[dim]Date: {date}  Universes: {', '.join(universes)}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    profile = _load_profile()
    config = _build_config(profile)
    max_workers = profile.get("max_workers", 3)

    positions = load_positions()
    open_tickers = sorted(t for t, p in positions.items() if p.get("status") == "open")
    if open_tickers:
        console.print(f"[dim]Currently open positions:[/dim] {', '.join(open_tickers)}")

    console.print(Rule("[bold]Unified scan[/bold]"))
    if dry_run:
        console.print("[yellow]DRY RUN — would scan universes + open positions, then preview opens/closes.[/yellow]")
        console.print(f"Universes: {', '.join(universes)}  Top {top_n} each")
        console.print(f"Carried-forward open positions: {', '.join(open_tickers) or '(none)'}")
        return

    scan = run_unified_scan(
        date=date,
        universes=universes,
        top_n=top_n,
        extra_tickers={"channel_open": open_tickers} if open_tickers else None,
        config=config,
        max_workers=max_workers,
    )
    console.print(f"Scanned [green]{len(scan['tags'])}[/green] unique ticker(s)")

    # Fetch live prices for ALL tickers that need P&L tracking (open + newly opened + closing)
    all_tracked = list(scan["tags"].keys())
    all_prices = get_current_prices(all_tracked) if all_tracked else {}

    opened, closed, still_open = update_positions(
        date, scan["results"], scan["tags"], prices=all_prices
    )
    console.print(
        f"Opened: [green]{len(opened)}[/green]  Closed: [red]{len(closed)}[/red]  "
        f"Unchanged open: [dim]{len(still_open)}[/dim]"
    )

    secondary_results = scan.get("secondary_results", {})
    primary_model = scan.get("primary_model", "")
    secondary_model = scan.get("secondary_model", "")

    positions_file = load_positions()

    def _enrich(entry: dict):
        t = entry["ticker"]
        result = scan["results"].get(t, {})
        # Use LLM-extracted price levels when available, otherwise fall back to
        # the entry_price stored in broadcast_positions.json when the BUY fired.
        pos_state = positions_file.get(t, {})
        entry["entry_price"] = (result.get("entry_price")
                                or entry.get("entry_price")
                                or pos_state.get("entry_price"))
        entry["stop_loss"] = result.get("stop_loss")
        entry["price_target"] = result.get("price_target")
        entry["bull_thesis"] = result.get("bull_thesis", "")
        entry["bear_concern"] = result.get("bear_concern", "")
        entry["judge_verdict"] = result.get("judge_verdict", "")
        entry["executive_summary"] = result.get("executive_summary", "")
        if secondary_results:
            sec = secondary_results.get(t, {})
            entry["secondary_rating"] = sec.get("rating", "")
            entry["secondary_model"] = secondary_model
            entry["primary_model"] = primary_model

    # Enrich all entries with LLM-extracted levels + reasoning
    # (current_price already set by update_positions via all_prices)
    for entry in opened:
        if not entry.get("current_price"):
            entry["current_price"] = all_prices.get(entry["ticker"])
        _enrich(entry)

    for entry in closed:
        if not entry.get("current_price"):
            entry["current_price"] = all_prices.get(entry["ticker"])
        _enrich(entry)

    # Enrich still_open positions with current price + days held for the daily digest
    if still_open:
        for entry in still_open:
            t = entry["ticker"]
            if not entry.get("current_price"):
                entry["current_price"] = all_prices.get(t)
            pos = positions_file.get(t, {})
            opened_date = entry.get("opened_date") or pos.get("opened_date", "")
            if opened_date:
                from datetime import datetime as _dt
                days = (_dt.fromisoformat(date) - _dt.fromisoformat(opened_date)).days
                entry["opened_date"] = opened_date
                entry["days_held"] = days
            _enrich(entry)  # _enrich now reads entry_price from pos_state as fallback

    # Send BUY/SELL event message
    message = format_position_update_message(date, opened, closed)
    if message:
        if send_telegram_broadcast(message):
            console.print("[green]BUY/SELL broadcast sent.[/green]")
        else:
            console.print("[yellow]BUY/SELL broadcast not sent (Telegram not configured or failed).[/yellow]")

        # Charts disabled — enable when ready
        # for entry in sorted(opened, key=lambda e: e["ticker"]):
        #     chart = generate_buy_chart(entry["ticker"], date, entry.get("entry_price"))
        #     if chart:
        #         send_telegram_photo(chart, caption=f"*{entry['ticker']}* — BUY {date}", broadcast=True)
    else:
        console.print("[dim]No new BUY/SELL events today.[/dim]")

    # Enrich newly opened positions with Day 1 metadata for the digest
    positions_state = load_positions()
    for entry in opened:
        t = entry["ticker"]
        entry.setdefault("opened_date", date)
        entry.setdefault("days_held", 0)

    # Always send the daily "still holding" digest if there are open positions
    if still_open or opened:
        all_open = list(still_open) + list(opened)
        positions_state = load_positions()
        stats = get_stats(positions_state)
        digest = format_holdings_digest(date, all_open, stats=stats)
        if send_telegram_broadcast(digest):
            console.print(f"[green]Holdings digest sent ({len(all_open)} open position(s)).[/green]")
    elif not opened and not closed:
        console.print("[dim]No open positions and no new signals — nothing to broadcast.[/dim]")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Broadcast BUY/SELL events to the Telegram channel")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--universes",
        default="full,sp500,nasdaq100",
        help="Comma-separated universes to scan (default: full,sp500,nasdaq100)",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of candidates per universe (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without LLM analysis or sending")
    args = parser.parse_args()

    universes = [u.strip() for u in args.universes.split(",") if u.strip()]
    run(args.date, universes, args.top, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
