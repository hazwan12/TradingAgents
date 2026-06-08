#!/usr/bin/env python3
"""Nightly analysis runner — reads user profile and stages orders for next market open.

Usage:
    python scripts/nightly_analysis.py              # use today's date
    python scripts/nightly_analysis.py --date 2026-06-02
    python scripts/nightly_analysis.py --dry-run    # print plan, don't run analysis

Schedule with cron (runs Mon-Fri at 3:55 PM ET):
    55 15 * * 1-5  cd /path/to/TradingAgents && python scripts/nightly_analysis.py
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.portfolio_advisor import PortfolioAdvisor
from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

console = Console()

TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
PROFILE_PATH = TRADINGAGENTS_HOME / "profile.json"
PORTFOLIO_CONFIG_PATH = TRADINGAGENTS_HOME / "portfolio_config.json"
ORDERS_DIR = TRADINGAGENTS_HOME / "orders"


def _load_profile() -> dict:
    if not PROFILE_PATH.exists():
        console.print(
            "[bold red]No profile found.[/bold red] "
            "Run [bold]python scripts/create_profile.py[/bold] first."
        )
        sys.exit(1)
    return json.loads(PROFILE_PATH.read_text())


def _load_portfolio() -> dict:
    if not PORTFOLIO_CONFIG_PATH.exists():
        console.print(
            "[bold red]No portfolio config found.[/bold red] "
            "Run [bold]python scripts/create_profile.py[/bold] first."
        )
        sys.exit(1)
    return json.loads(PORTFOLIO_CONFIG_PATH.read_text())


def _build_config(profile: dict) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = profile.get("llm_provider", config["llm_provider"])
    config["deep_think_llm"] = profile.get("deep_think_llm", config["deep_think_llm"])
    config["quick_think_llm"] = profile.get("quick_think_llm", config["quick_think_llm"])
    return config


def _calc_limit_price(estimated_price: float, direction: str, buffer_pct: float) -> float:
    """Buy limit: slightly above estimated to ensure fill. Sell limit: slightly below."""
    if direction == "buy":
        return round(estimated_price * (1 + buffer_pct / 100), 4)
    return round(estimated_price * (1 - buffer_pct / 100), 4)


def _recommendations_to_orders(
    recommendations: list[dict], date: str, buffer_pct: float = 0.5
) -> dict:
    orders = []
    for rec in recommendations:
        if rec.get("units_to_trade", 0) == 0:
            continue
        direction = "buy" if rec["units_to_trade"] > 0 else "sell"
        estimated_price = rec["current_price"]
        limit_price = _calc_limit_price(estimated_price, direction, buffer_pct)
        orders.append(
            {
                "id": str(uuid.uuid4()),
                "ticker": rec["ticker"],
                "action": rec["action"],
                "rating": rec.get("rating", ""),
                "units": abs(rec["units_to_trade"]),
                "direction": direction,
                "estimated_price": estimated_price,
                "limit_price": limit_price,
                "limit_price_buffer_pct": buffer_pct,
                "estimated_total": round(limit_price * abs(rec["units_to_trade"]), 2),
                "order_type": "limit",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "executed_at": None,
                "executed_price": None,
                "executed_units": None,
                "notes": "",
            }
        )
    return {
        "date": date,
        "generated_at": datetime.now().isoformat(),
        "orders": orders,
    }


def _write_orders(orders_doc: dict, date: str) -> Path:
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    path = ORDERS_DIR / f"ORDERS_{date}.json"
    path.write_text(json.dumps(orders_doc, indent=2))
    return path


def _print_orders_table(orders: list[dict]):
    if not orders:
        console.print("[yellow]No actionable orders generated (all positions are Hold).[/yellow]")
        return

    t = Table(
        title="Staged Orders — review before market open",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    t.add_column("Ticker", style="bold white", width=8)
    t.add_column("Direction", width=6)
    t.add_column("Rating", width=12)
    t.add_column("Units", justify="right", width=7)
    t.add_column("Est. Price", justify="right", width=10)
    t.add_column("Limit Price", justify="right", width=11)
    t.add_column("Est. Total", justify="right", width=11)
    t.add_column("Status", width=8)

    for order in sorted(orders, key=lambda o: o["ticker"]):
        direction_style = "green" if order["direction"] == "buy" else "red"
        t.add_row(
            order["ticker"],
            f"[{direction_style}]{order['direction'].upper()}[/{direction_style}]",
            order.get("rating", ""),
            f"{order['units']:.2f}",
            f"${order['estimated_price']:,.2f}",
            f"[bold]${order['limit_price']:,.4f}[/bold]",
            f"${order['estimated_total']:,.2f}",
            f"[yellow]{order['status']}[/yellow]",
        )

    console.print()
    console.print(t)


def _load_ticker_file(path: str) -> list[str]:
    """Load tickers from a screener JSON or a plain text file (one per line)."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Ticker file not found: {path}[/red]")
        sys.exit(1)
    if p.suffix == ".json":
        doc = json.loads(p.read_text())
        tickers = doc.get("tickers", [])
    else:
        tickers = [line.strip().upper() for line in p.read_text().splitlines() if line.strip()]
    if not tickers:
        console.print(f"[red]No tickers found in {path}[/red]")
        sys.exit(1)
    return tickers


def run(
    date: str,
    dry_run: bool = False,
    ticker_file: str | None = None,
    tickers_override: list[str] | None = None,
):
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]TradingAgents — Nightly Analysis[/bold cyan]\n"
            f"[dim]Date: {date}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    profile = _load_profile()
    portfolio = _load_portfolio()

    # Ticker source priority: tickers_override > --ticker-file > portfolio config
    if tickers_override is not None:
        tickers = tickers_override
        console.print(f"[dim]Tickers from screener:[/dim] {len(tickers)} tickers")
    elif ticker_file:
        tickers = _load_ticker_file(ticker_file)
        console.print(f"[dim]Tickers from file:[/dim] {ticker_file} ({len(tickers)} tickers)")
    else:
        tickers = portfolio.get("tickers", [])

    budget = portfolio.get("budget", 0.0)
    holdings = portfolio.get("holdings", {})
    max_workers = portfolio.get("max_workers", profile.get("max_workers", 3))

    console.print(f"[dim]Tickers:[/dim] {', '.join(tickers)}")
    console.print(f"[dim]Budget:[/dim]  ${budget:,.2f}   [dim]Holdings:[/dim] {len(holdings)} positions")
    console.print()

    if dry_run:
        console.print("[bold yellow]DRY RUN — skipping analysis. Would scan:[/bold yellow]")
        for t in tickers:
            console.print(f"  {t}")
        return

    console.print(Rule("[bold]Running analysis...[/bold]"))

    config = _build_config(profile)
    advisor = PortfolioAdvisor(
        budget=budget,
        holdings=holdings,
        max_workers=max_workers,
        config=config,
    )

    result = advisor.advise(tickers, date)

    if not result.get("success"):
        console.print(f"[bold red]Analysis failed:[/bold red] {result.get('error', 'unknown error')}")
        sys.exit(1)

    console.print()
    console.print(Rule("[bold]Results[/bold]"))
    console.print(f"Portfolio value: [bold]${result['portfolio_value']:,.2f}[/bold]")
    console.print(f"Cash available:  [bold]${result['cash_available']:,.2f}[/bold]")
    console.print(
        f"Buy power needed: [green]${result['total_buy_power_needed']:,.2f}[/green]   "
        f"Sell proceeds:  [red]${result['total_sell_proceeds']:,.2f}[/red]"
    )

    # Stage orders
    buffer_pct = profile.get("limit_price_buffer_pct", 0.5)
    orders_doc = _recommendations_to_orders(result["recommendations"], date, buffer_pct)
    _print_orders_table(orders_doc["orders"])

    orders_path = _write_orders(orders_doc, date)

    console.print()
    console.print(Rule("[bold green]Done[/bold green]"))
    console.print(f"Staged orders: [cyan]{orders_path}[/cyan]")
    console.print(f"CSV report:    [cyan]{result['summary_csv']}[/cyan]")
    console.print(f"MD report:     [cyan]{result['summary_md']}[/cyan]")
    console.print()
    console.print("[dim]When your trades execute, run:[/dim]")
    console.print(f"  [bold]python scripts/update_holdings.py[/bold]")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Run nightly portfolio analysis")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without running analysis",
    )
    parser.add_argument(
        "--ticker-file",
        metavar="FILE",
        help="JSON (from pre_screener.py) or .txt (one ticker per line) to use instead of portfolio config tickers",
    )
    args = parser.parse_args()
    run(args.date, dry_run=args.dry_run, ticker_file=args.ticker_file)


if __name__ == "__main__":
    main()
