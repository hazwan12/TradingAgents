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
    python scripts/daily.py --run              # non-interactive: full pipeline (full+sp500+nasdaq100, top 20 each)
    python scripts/daily.py --run --universe full --top 30
    python scripts/daily.py --run --universe sp500,nasdaq100
    python scripts/daily.py --run --skip-sync  # skip Moomoo sync step
    python scripts/daily.py --pipeline         # interactive full pipeline (prompts for universe(s))
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

from tradingagents.default_config import DEFAULT_CONFIG

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


def _build_secondary_config(profile: dict) -> dict | None:
    """Build config for the secondary model if configured in profile, else None."""
    secondary_model = profile.get("secondary_deep_think_llm", "").strip()
    if not secondary_model:
        return None
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = profile.get("secondary_llm_provider", profile.get("llm_provider", config["llm_provider"]))
    config["deep_think_llm"] = secondary_model
    config["quick_think_llm"] = profile.get("quick_think_llm", config["quick_think_llm"])
    return config


def _run_unified_scan(
    date: str,
    universes: list[str],
    top_n: int,
    held_tickers: set[str],
    force_refresh: bool = False,
) -> dict | None:
    """Run (or reuse cached) unified scan across all requested universes + held tickers."""
    from scripts.market_scan import run_unified_scan
    from scripts.nightly_analysis import _build_config
    _log(f"Scanning universes=[cyan]{', '.join(universes)}[/cyan]  top={top_n} each …")
    profile = json.loads(PROFILE_PATH.read_text())
    primary_config = _build_config(profile)
    secondary_config = _build_secondary_config(profile)
    if secondary_config:
        _log(f"Dual model: primary=[cyan]{primary_config.get('deep_think_llm')}[/cyan]  secondary=[cyan]{secondary_config['deep_think_llm']}[/cyan]")
    else:
        _log(f"Model: [cyan]{primary_config.get('deep_think_llm')}[/cyan]")
    scan = run_unified_scan(
        date=date,
        universes=universes,
        top_n=top_n,
        extra_tickers={"held": sorted(held_tickers)} if held_tickers else None,
        config=primary_config,
        secondary_config=secondary_config,
        force_refresh=force_refresh,
    )
    n_candidates = len(scan["tags"])
    _log(f"Unified scan covers [green]{n_candidates}[/green] ticker(s) across {len(universes)} universe(s)")
    return scan


def _run_unified_scan_interactive(date: str, held_tickers: set[str], force_refresh: bool = False) -> dict | None:
    """Prompt for universes / top-N, then run the unified scan."""
    console.print()
    universes = questionary.checkbox(
        "Which universe(s) to scan? (space to toggle, enter to confirm)",
        choices=[
            questionary.Choice(label, value=src, checked=True)
            for src, label in _UNIVERSE_LABELS.items()
            if src in ("full", "sp500", "nasdaq100")
        ],
    ).ask()
    if not universes:
        return None

    top_raw = questionary.text(
        "How many top candidates to keep per universe?",
        default="20",
        validate=lambda v: (v.isdigit() and int(v) >= 1) or "Enter a positive number",
    ).ask()
    if top_raw is None:
        return None

    if "full" in universes:
        console.print("[yellow]Scanning full US market — this takes a few minutes.[/yellow]")

    return _run_unified_scan(date, universes, int(top_raw), held_tickers, force_refresh=force_refresh)


def _run_analysis(tickers: list[str], date: str, precomputed_scan: dict | None = None):
    """Run nightly_analysis with the given ticker list."""
    from scripts.nightly_analysis import run as run_analysis
    preview = ", ".join(tickers[:8]) + (f" … +{len(tickers) - 8} more" if len(tickers) > 8 else "")
    _log(f"Analysing [cyan]{len(tickers)}[/cyan] ticker(s): {preview}")
    run_analysis(date=date, ticker_file=None, tickers_override=tickers, precomputed_scan=precomputed_scan)


def _run_fills():
    from scripts.update_holdings import run as run_fills
    run_fills()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_full_pipeline(
    date: str,
    non_interactive: bool = False,
    universes: list[str] | None = None,
    top_n: int = 20,
    skip_sync: bool = False,
    moomoo_host: str = "127.0.0.1",
    moomoo_port: int = 11111,
    force: bool = False,
    force_refresh: bool = False,
):
    universes = universes or ["full", "sp500", "nasdaq100"]
    today = datetime.now().strftime("%Y-%m-%d")
    if not force and date == today and _us_market_is_open():
        _warn_market_open(non_interactive)

    total_steps = (3 if skip_sync else 4)
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

    # Step 2: Unified scan (all requested universes, shared with the channel broadcast)
    step += 1
    with _step("Market screener", step, total_steps):
        if non_interactive:
            scan = _run_unified_scan(date, universes, top_n, held_tickers, force_refresh=force_refresh)
        else:
            scan = _run_unified_scan_interactive(date, held_tickers, force_refresh=force_refresh)

    if not scan or not scan.get("tags"):
        tickers = portfolio.get("tickers", [])
        if not tickers:
            console.print("[red]No tickers available. Aborting.[/red]")
            return
        _log(f"Falling back to {len(tickers)} tickers from portfolio config.")
        tickers = sorted(set(tickers) | held_tickers)
        precomputed_scan = None
    else:
        tickers = sorted(set(scan["tags"].keys()) | held_tickers)
        precomputed_scan = scan["results"]

    # Step 3: LLM analysis + order staging
    step += 1
    with _step("LLM analysis & order staging", step, total_steps):
        _run_analysis(tickers, date, precomputed_scan=precomputed_scan)

    # Step 4: Channel broadcast (reuses cached scan — no extra LLM cost)
    step += 1
    with _step("Channel broadcast", step, total_steps):
        from scripts.broadcast_top10 import run as run_broadcast
        run_broadcast(date=date, universes=universes, top_n=top_n)

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


def mode_sync(moomoo_host: str = "127.0.0.1", moomoo_port: int = 11111):
    """Step 1 only — sync Moomoo holdings into portfolio_config.json."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Step 1: Moomoo Sync[/bold cyan]",
            border_style="cyan",
        )
    )
    _require_profile()
    with _step("Sync Moomoo holdings", 1, 1):
        _sync_moomoo(moomoo_host, moomoo_port)
    console.print("[bold green]Sync complete.[/bold green]")
    console.print("[dim]Next: python scripts/daily.py --scan[/dim]")
    console.print()


def mode_scan(
    date: str,
    universes: list[str] | None = None,
    top_n: int = 20,
    force_refresh: bool = False,
):
    """Step 2 only — run unified market scan and save to cache.
    Resumes automatically from any prior checkpoint for this date."""
    universes = universes or ["full", "sp500", "nasdaq100"]
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Step 2: Market Scan[/bold cyan]\n"
            f"[dim]Date: {date}  Universes: {', '.join(universes)}  Top: {top_n}[/dim]",
            border_style="cyan",
        )
    )
    _require_profile()
    portfolio = json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    held_tickers = set(portfolio.get("holdings", {}).keys())

    with _step("Unified market scan", 1, 1):
        scan = _run_unified_scan(date, universes, top_n, held_tickers, force_refresh=force_refresh)

    n_results = len(scan.get("results", {})) if scan else 0
    n_total = len(scan.get("tags", {})) if scan else 0
    console.print(f"[bold green]Scan complete:[/bold green] {n_results}/{n_total} tickers analyzed")
    console.print(f"[dim]Cache: {_cache_path(date)}[/dim]")
    console.print("[dim]Next: python scripts/daily.py --advise[/dim]")
    console.print()


def mode_advise(date: str):
    """Step 3 only — run personal portfolio advisor using cached scan.
    Requires a prior --scan run for this date."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Step 3: Personal Advisor[/bold cyan]\n"
            f"[dim]Date: {date}[/dim]",
            border_style="cyan",
        )
    )
    _require_profile()
    from scripts.market_scan import _cache_path, _load_cache
    cache = _load_cache(date)
    if not cache or not cache.get("results"):
        console.print(
            f"[bold red]No cached scan found for {date}.[/bold red] "
            "Run [bold]python scripts/daily.py --scan[/bold] first."
        )
        return

    portfolio = json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    held_tickers = set(portfolio.get("holdings", {}).keys())
    tickers = sorted(set(cache["tags"].keys()) | held_tickers)
    precomputed_scan = cache["results"]

    n_results = len(precomputed_scan)
    console.print(f"[dim]Using cached scan: {n_results} rated ticker(s)[/dim]")

    with _step("Personal advisor & order staging", 1, 1):
        _run_analysis(tickers, date, precomputed_scan=precomputed_scan)

    console.print("[dim]Next: review orders, then python scripts/daily.py --fills[/dim]")
    console.print()


def _cache_path(date: str):
    from pathlib import Path
    return Path("reports") / f"UNIFIED_SCAN_{date}.json"


def mode_broadcast(date: str, universes: list[str] | None = None, top_n: int = 20):
    """Step 4 only — run channel broadcast using cached scan."""
    universes = universes or ["full", "sp500", "nasdaq100"]
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Step 4: Channel Broadcast[/bold cyan]\n"
            f"[dim]Date: {date}[/dim]",
            border_style="cyan",
        )
    )
    from scripts.broadcast_top10 import run as run_broadcast
    with _step("Channel broadcast", 1, 1):
        run_broadcast(date=date, universes=universes, top_n=top_n)


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
    group.add_argument(
        "--sync",
        action="store_true",
        help="Step 1 only: sync Moomoo holdings into portfolio_config.json",
    )
    group.add_argument(
        "--scan",
        action="store_true",
        help="Step 2 only: run unified market scan and save to cache (resumes from checkpoint if interrupted)",
    )
    group.add_argument(
        "--advise",
        action="store_true",
        help="Step 3 only: run personal portfolio advisor using cached scan from --scan",
    )
    group.add_argument(
        "--broadcast",
        action="store_true",
        help="Step 4 only: run channel broadcast using cached scan (reuses --scan results, no extra LLM cost)",
    )

    screener = parser.add_argument_group("screener options (--run only)")
    screener.add_argument(
        "--universe",
        default="full,sp500,nasdaq100",
        metavar="UNIVERSE",
        help=(
            "Comma-separated universe(s) to scan in one unified pass "
            "(default: full,sp500,nasdaq100). Options: full, sp500, nasdaq100."
        ),
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
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "Ignore any cached unified scan for this date and re-run the LLM analysis "
            "from scratch (e.g. after changing TRADINGAGENTS_DEEP_THINK_LLM to compare models)"
        ),
    )

    args = parser.parse_args()
    _require_profile()

    if args.run:
        mode_full_pipeline(
            date=args.date,
            non_interactive=True,
            universes=[u.strip() for u in args.universe.split(",") if u.strip()],
            top_n=args.top,
            skip_sync=args.skip_sync,
            moomoo_host=args.moomoo_host,
            moomoo_port=args.moomoo_port,
            force=args.force,
            force_refresh=args.force_refresh,
        )
    elif args.pipeline:
        mode_full_pipeline(
            date=args.date,
            non_interactive=False,
            skip_sync=args.skip_sync,
            moomoo_host=args.moomoo_host,
            moomoo_port=args.moomoo_port,
            force=args.force,
            force_refresh=args.force_refresh,
        )
    elif args.analyse:
        mode_analyse_only(args.date)
    elif args.fills:
        mode_record_fills()
    elif args.sync:
        mode_sync(moomoo_host=args.moomoo_host, moomoo_port=args.moomoo_port)
    elif args.scan:
        mode_scan(
            date=args.date,
            universes=[u.strip() for u in args.universe.split(",") if u.strip()],
            top_n=args.top,
            force_refresh=args.force_refresh,
        )
    elif args.advise:
        mode_advise(date=args.date)
    elif args.broadcast:
        mode_broadcast(
            date=args.date,
            universes=[u.strip() for u in args.universe.split(",") if u.strip()],
            top_n=args.top,
        )
    else:
        _interactive_menu(args.date)


if __name__ == "__main__":
    main()
