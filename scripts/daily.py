#!/usr/bin/env python3
"""Single entrypoint for the daily trading workflow.

Menu
----
  [1] Full pipeline   — sync Moomoo holdings → screen market → analyse → stage orders
  [2] Analyse only    — skip screener, use saved tickers from portfolio config
  [3] Record fills    — update holdings after broker executes staged orders
  [4] Exit

Usage
-----
    python scripts/daily.py                    # interactive menu
    python scripts/daily.py --run              # non-interactive: full pipeline (sp500, top 20)
    python scripts/daily.py --run --universe full --top 30
    python scripts/daily.py --run --skip-sync  # skip Moomoo sync step
    python scripts/daily.py --pipeline         # interactive full pipeline (prompts for universe)
    python scripts/daily.py --analyse          # non-interactive: analyse only
    python scripts/daily.py --fills            # non-interactive: record fills
    python scripts/daily.py --date 2026-06-04
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, time as dtime
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
# Market-hours guard
# ---------------------------------------------------------------------------

_MARKET_OPEN  = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)


def _us_market_is_open() -> bool:
    """Return True if the US equity market is currently open (9:30–16:00 ET, Mon–Fri)."""
    try:
        try:
            from zoneinfo import ZoneInfo
            ET = ZoneInfo("America/New_York")
        except Exception:
            import pytz
            ET = pytz.timezone("America/New_York")
        now_et = datetime.now(ET)
    except Exception:
        return False  # can't determine timezone — skip the check

    if now_et.weekday() >= 5:  # Saturday / Sunday
        return False
    t = now_et.time().replace(tzinfo=None)
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def _warn_market_open(non_interactive: bool):
    """Print a market-hours warning. Abort if non-interactive, prompt if interactive."""
    console.print()
    console.print(
        Panel(
            "[bold red]US market is currently open (9:30–16:00 ET)[/bold red]\n\n"
            "Running the pipeline now produces unreliable results:\n"
            "  • yfinance returns incomplete intraday bars — LLM signals are degraded\n"
            "  • Limit prices will be stale by the time you review the orders\n\n"
            "[bold yellow]Recommended:[/bold yellow] run after market close (4:00 PM ET / 4:00 AM SGT)\n"
            "and submit orders before the next open.",
            border_style="red",
            title="[bold red]Market Hours Warning[/bold red]",
        )
    )
    console.print()

    if non_interactive:
        console.print(
            "[red]Aborting.[/red] Pass [bold]--force[/bold] to override this check."
        )
        sys.exit(1)

    confirmed = questionary.confirm(
        "Run anyway? (orders generated now should NOT be executed)", default=False
    ).ask()
    if not confirmed:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


@contextmanager
def _step(label: str, step_num: int, total_steps: int):
    """Print a step header on entry and elapsed time on exit."""
    console.print()
    console.print(Rule(f"[bold cyan]Step {step_num}/{total_steps}  —  {label}[/bold cyan]"))
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        mins, secs = divmod(int(elapsed), 60)
        duration = f"{mins}m {secs}s" if mins else f"{secs}s"
        console.print(f"[dim]  ✓  {label}  ({duration})[/dim]")


def _log(msg: str):
    """Timestamped progress line."""
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim]  {msg}")


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


def _sync_moomoo(host: str, port: int):
    """Sync live Moomoo positions into portfolio_config.json (auto-confirm)."""
    from scripts.sync_moomoo_holdings import run as _moomoo_run
    _log(f"Connecting to OpenD at [cyan]{host}:{port}[/cyan] …")
    _moomoo_run(host=host, port=port, dry_run=False, yes=True, paper=False)


def _run_screener(date: str, universe: str, top_n: int) -> list[str] | None:
    """Non-interactive screener run."""
    from scripts.pre_screener import screen
    _log(f"Screening universe=[cyan]{universe}[/cyan]  top={top_n} …")
    result = screen(top_n=top_n, universe_source=universe, date=date)
    if not result.get("success"):
        console.print("[red]Screener failed — will fall back to portfolio tickers.[/red]")
        return None
    tickers = result.get("candidates", [])
    _log(f"Screener selected [green]{len(tickers)}[/green] candidates")
    return tickers


def _run_screener_interactive(date: str) -> list[str] | None:
    """Prompt for universe / top-N, then run screener."""
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
        console.print("[yellow]Scanning full US market — this takes a few minutes.[/yellow]")

    _log(f"Screening universe=[cyan]{universe}[/cyan]  top={top_raw} …")
    result = screen(top_n=int(top_raw), universe_source=universe, date=date)
    if not result.get("success"):
        console.print("[red]Screener failed — will fall back to portfolio tickers.[/red]")
        return None
    tickers = result.get("candidates", [])
    _log(f"Screener selected [green]{len(tickers)}[/green] candidates")
    return tickers


def _run_analysis(tickers: list[str], date: str):
    """Run nightly_analysis with the given ticker list."""
    from scripts.nightly_analysis import run as run_analysis
    preview = ", ".join(tickers[:8]) + (f" … +{len(tickers) - 8} more" if len(tickers) > 8 else "")
    _log(f"Analysing [cyan]{len(tickers)}[/cyan] ticker(s): {preview}")
    run_analysis(date=date, ticker_file=None, tickers_override=tickers)


def _run_fills():
    from scripts.update_holdings import run as run_fills
    run_fills()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_full_pipeline(
    date: str,
    non_interactive: bool = False,
    universe: str = "full",
    top_n: int = 20,
    skip_sync: bool = False,
    moomoo_host: str = "127.0.0.1",
    moomoo_port: int = 11111,
    force: bool = False,
):
    if not force and _us_market_is_open():
        _warn_market_open(non_interactive)

    total_steps = (2 if skip_sync else 3)
    step = 0
    pipeline_start = time.monotonic()

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Full Pipeline[/bold cyan]\n"
            f"[dim]Date: {date}  |  Steps: {total_steps}[/dim]",
            border_style="cyan",
        )
    )

    # Step 1 (optional): Moomoo sync
    if not skip_sync:
        step += 1
        with _step("Sync Moomoo holdings", step, total_steps):
            _sync_moomoo(moomoo_host, moomoo_port)

    # Reload portfolio (may have been updated by sync)
    portfolio = json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    held_tickers = set(portfolio.get("holdings", {}).keys())

    # Step 2: Screener
    step += 1
    with _step("Market screener", step, total_steps):
        if non_interactive:
            tickers = _run_screener(date, universe, top_n)
        else:
            tickers = _run_screener_interactive(date)

    if not tickers:
        tickers = portfolio.get("tickers", [])
        if not tickers:
            console.print("[red]No tickers available. Aborting.[/red]")
            return
        _log(f"Falling back to {len(tickers)} tickers from portfolio config.")

    # Always include held positions so they get a fresh signal
    extra = held_tickers - set(tickers)
    if extra:
        _log(f"Adding {len(extra)} held ticker(s) not in screener: {', '.join(sorted(extra))}")
    tickers = sorted(set(tickers) | held_tickers)

    # Step 3: LLM analysis + order staging
    step += 1
    with _step("LLM analysis & order staging", step, total_steps):
        _run_analysis(tickers, date)

    # Final summary
    total_elapsed = time.monotonic() - pipeline_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    total_duration = f"{total_mins}m {total_secs}s" if total_mins else f"{total_secs}s"
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Pipeline complete[/bold green]  —  {total_duration} total\n"
            "[dim]Review staged orders, execute in Moomoo, then run "
            "[bold]python scripts/daily.py --fills[/bold][/dim]",
            border_style="green",
        )
    )
    console.print()


def mode_analyse_only(date: str):
    console.print()
    console.print(Rule("[bold cyan]Analyse Only — Using Portfolio Tickers[/bold cyan]"))

    portfolio = json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    tickers = portfolio.get("tickers", [])
    if not tickers:
        console.print("[red]No tickers in portfolio config. Run create_profile.py first.[/red]")
        return

    preview = ", ".join(tickers[:10]) + (f" … +{len(tickers) - 10} more" if len(tickers) > 10 else "")
    console.print(f"[dim]Tickers ({len(tickers)}):[/dim] {preview}")
    with _step("LLM analysis & order staging", 1, 1):
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
                "Full pipeline  — sync holdings → screen market → analyse → stage orders",
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
        mode_full_pipeline(date, non_interactive=False)
    elif choice == "analyse":
        mode_analyse_only(date)
    elif choice == "fills":
        mode_record_fills()


def main():
    parser = argparse.ArgumentParser(
        description="Daily TradingAgents workflow — sync, screen, analyse, and record fills"
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date YYYY-MM-DD (default: today)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run",
        action="store_true",
        help="Non-interactive full pipeline: sync → screen → analyse → stage orders",
    )
    group.add_argument(
        "--pipeline",
        action="store_true",
        help="Interactive full pipeline (prompts for universe/top-N)",
    )
    group.add_argument(
        "--analyse",
        action="store_true",
        help="Non-interactive: analyse portfolio tickers only (no sync, no screener)",
    )
    group.add_argument(
        "--fills",
        action="store_true",
        help="Non-interactive: record broker fills",
    )

    screener = parser.add_argument_group("screener options (--run only)")
    screener.add_argument(
        "--universe",
        default="full",
        choices=list(_UNIVERSE_LABELS.keys()),
        metavar="UNIVERSE",
        help="Universe to screen: full (default), sp500, nasdaq100, sp500+nasdaq100",
    )
    screener.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Top-N candidates to keep from screener (default: 20)",
    )

    moomoo = parser.add_argument_group("Moomoo options")
    moomoo.add_argument("--moomoo-host", default="127.0.0.1", metavar="HOST", help="OpenD host (default: 127.0.0.1)")
    moomoo.add_argument("--moomoo-port", type=int, default=11111, metavar="PORT", help="OpenD port (default: 11111)")
    moomoo.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip the Moomoo holdings sync step",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the market-hours safety check (orders generated during market hours should NOT be executed)",
    )

    args = parser.parse_args()
    _require_profile()

    if args.run:
        mode_full_pipeline(
            date=args.date,
            non_interactive=True,
            universe=args.universe,
            top_n=args.top,
            skip_sync=args.skip_sync,
            moomoo_host=args.moomoo_host,
            moomoo_port=args.moomoo_port,
            force=args.force,
        )
    elif args.pipeline:
        mode_full_pipeline(
            date=args.date,
            non_interactive=False,
            skip_sync=args.skip_sync,
            moomoo_host=args.moomoo_host,
            moomoo_port=args.moomoo_port,
            force=args.force,
        )
    elif args.analyse:
        mode_analyse_only(args.date)
    elif args.fills:
        mode_record_fills()
    else:
        _interactive_menu(args.date)


if __name__ == "__main__":
    main()
