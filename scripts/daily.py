#!/usr/bin/env python3
"""Single entrypoint for the daily trading workflow.

Menu
----
  [1] Full pipeline   — screen market → analyse → stage orders
  [2] Analyse only    — skip screener, use saved tickers from portfolio config
  [3] Record fills    — update holdings after broker executes staged orders
  [4] Exit

Usage
-----
    python scripts/daily.py              # interactive menu
    python scripts/daily.py --pipeline  # non-interactive: full pipeline
    python scripts/daily.py --analyse   # non-interactive: analyse only
    python scripts/daily.py --fills     # non-interactive: record fills
    python scripts/daily.py --date 2026-06-04
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
import questionary

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
PROFILE_PATH = TRADINGAGENTS_HOME / "profile.json"
PORTFOLIO_CONFIG_PATH = TRADINGAGENTS_HOME / "portfolio_config.json"

_UNIVERSE_LABELS = {
    "full":            "All US-listed stocks via NASDAQ FTP (~9,000) — catches emerging tickers",
    "sp500":           "S&P 500 constituents (~500)",
    "nasdaq100":       "Nasdaq 100 constituents (~100)",
    "sp500+nasdaq100": "S&P 500 + Nasdaq 100 union (~550)",
}


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_profile():
    if not PROFILE_PATH.exists() or not PORTFOLIO_CONFIG_PATH.exists():
        console.print(
            "[bold red]No profile found.[/bold red] "
            "Run [bold]python scripts/create_profile.py[/bold] first."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _run_screener(date: str) -> list[str] | None:
    """Ask which universe, run pre_screener.screen(), return ranked tickers."""
    from scripts.pre_screener import screen

    console.print()
    universe = questionary.select(
        "Which universe to scan?",
        choices=[
            questionary.Choice(label, value=src)
            for src, label in _UNIVERSE_LABELS.items()
        ],
    ).ask()
    if universe is None:
        return None

    top_raw = questionary.text(
        "How many top candidates to keep?",
        default="20",
        validate=lambda v: (v.isdigit() and int(v) >= 1) or "Enter a positive number",
    ).ask()
    if top_raw is None:
        return None

    if universe == "full":
        console.print(
            "[yellow]Scanning full US market — this takes a few minutes.[/yellow]"
        )

    result = screen(top_n=int(top_raw), universe_source=universe, date=date)
    if not result.get("success"):
        console.print("[red]Screener failed. Falling back to portfolio tickers.[/red]")
        return None

    tickers = result.get("candidates", [])
    console.print(f"\n[green]Screener selected {len(tickers)} candidates.[/green]")
    return tickers


def _run_analysis(tickers: list[str], date: str):
    """Run nightly_analysis.run() with the given ticker list."""
    from scripts.nightly_analysis import run as run_analysis
    run_analysis(date=date, ticker_file=None, tickers_override=tickers)


def _run_fills():
    """Run update_holdings.run() interactively."""
    from scripts.update_holdings import run as run_fills
    run_fills()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_full_pipeline(date: str):
    console.print()
    console.print(Rule("[bold cyan]Full Pipeline — Screen → Analyse → Stage Orders[/bold cyan]"))

    portfolio = json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    held_tickers = set(portfolio.get("holdings", {}).keys())

    tickers = _run_screener(date)
    if not tickers:
        # Screener failed or was cancelled — fall back to portfolio tickers
        tickers = portfolio.get("tickers", [])
        if not tickers:
            console.print("[red]No tickers available. Aborting.[/red]")
            return
        console.print(f"[dim]Using {len(tickers)} tickers from portfolio config.[/dim]")

    # Always include held positions so they get a fresh buy/hold/sell signal
    # even if they dropped off the screener's top-N today.
    new_tickers = held_tickers - set(tickers)
    if new_tickers:
        console.print(
            f"[dim]Adding {len(new_tickers)} held ticker(s) not in screener results: "
            f"{', '.join(sorted(new_tickers))}[/dim]"
        )
    tickers = sorted(set(tickers) | held_tickers)

    _run_analysis(tickers, date)


def mode_analyse_only(date: str):
    console.print()
    console.print(Rule("[bold cyan]Analyse Only — Using Portfolio Tickers[/bold cyan]"))

    portfolio = json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    tickers = portfolio.get("tickers", [])
    if not tickers:
        console.print("[red]No tickers in portfolio config. Run create_profile.py first.[/red]")
        return

    console.print(f"[dim]Tickers ({len(tickers)}):[/dim] {', '.join(tickers[:10])}"
                  + (f" … +{len(tickers)-10} more" if len(tickers) > 10 else ""))
    _run_analysis(tickers, date)


def mode_record_fills():
    console.print()
    console.print(Rule("[bold cyan]Record Fills — Update Holdings[/bold cyan]"))
    _run_fills()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _interactive_menu(date: str):
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Daily Workflow[/bold cyan]\n"
            f"[dim]Date: {date}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    choice = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice(
                "Full pipeline  — screen market → analyse → stage orders",
                value="pipeline",
            ),
            questionary.Choice(
                "Analyse only   — skip screener, use saved portfolio tickers",
                value="analyse",
            ),
            questionary.Choice(
                "Record fills   — update holdings after broker executes orders",
                value="fills",
            ),
            questionary.Choice("Exit", value="exit"),
        ],
    ).ask()

    if choice is None or choice == "exit":
        return

    if choice == "pipeline":
        mode_full_pipeline(date)
    elif choice == "analyse":
        mode_analyse_only(date)
    elif choice == "fills":
        mode_record_fills()


def main():
    parser = argparse.ArgumentParser(
        description="Daily TradingAgents workflow — screen, analyse, and record fills"
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date YYYY-MM-DD (default: today)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pipeline",
        action="store_true",
        help="Non-interactive: run full pipeline (screen → analyse → stage orders)",
    )
    group.add_argument(
        "--analyse",
        action="store_true",
        help="Non-interactive: analyse only using portfolio tickers",
    )
    group.add_argument(
        "--fills",
        action="store_true",
        help="Non-interactive: record broker fills",
    )
    args = parser.parse_args()

    _require_profile()

    if args.pipeline:
        mode_full_pipeline(args.date)
    elif args.analyse:
        mode_analyse_only(args.date)
    elif args.fills:
        mode_record_fills()
    else:
        _interactive_menu(args.date)


if __name__ == "__main__":
    main()
