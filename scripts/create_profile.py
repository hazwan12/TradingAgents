#!/usr/bin/env python3
"""Setup wizard — create user profile and initial portfolio config."""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box
import questionary

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
PROFILE_PATH = TRADINGAGENTS_HOME / "profile.json"
PORTFOLIO_CONFIG_PATH = TRADINGAGENTS_HOME / "portfolio_config.json"

PROVIDER_MODELS = {
    "google": {
        "deep": "gemini-2.5-flash",
        "quick": "gemini-2.5-flash",
        "note": "Free tier — use Gemini 2.5 Flash for both models",
    },
    "ollama": {
        "deep": "qwen2.5:14b",
        "quick": "qwen2.5:7b",
        "note": "Local inference — no API key required",
    },
    "openai": {
        "deep": "gpt-4o",
        "quick": "gpt-4o-mini",
        "note": "Requires OPENAI_API_KEY",
    },
    "anthropic": {
        "deep": "claude-opus-4-8",
        "quick": "claude-haiku-4-5-20251001",
        "note": "Requires ANTHROPIC_API_KEY",
    },
}

# Sentinel stored in profile when auto-detect is chosen.
_AUTO_PROVIDER = "auto"

# Keywords that trigger the pre-screener pipeline instead of manual entry.
# Maps user input → pre_screener.fetch_universe source name.
_UNIVERSE_KEYWORDS = {
    "all":              "full",
    "full":             "full",
    "sp500":            "sp500",
    "nasdaq100":        "nasdaq100",
    "sp500+nasdaq100":  "sp500+nasdaq100",
}

_UNIVERSE_LABELS = {
    "full":             "All US-listed stocks via NASDAQ FTP (~9,000) — catches emerging tickers",
    "sp500":            "S&P 500 constituents (~500)",
    "nasdaq100":        "Nasdaq 100 constituents (~100)",
    "sp500+nasdaq100":  "S&P 500 + Nasdaq 100 union (~550)",
}


def _print_header():
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents Setup Wizard[/bold cyan]\n"
            "[dim]Creates your user profile and initial portfolio config[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def _print_summary(profile: dict, portfolio: dict):
    console.print()
    console.print(Rule("[bold green]Profile Summary[/bold green]"))
    console.print()

    ticker_display = ", ".join(portfolio["tickers"][:10])
    if len(portfolio["tickers"]) > 10:
        ticker_display += f" … (+{len(portfolio['tickers']) - 10} more)"

    t = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    t.add_column("Key", style="bold dim", width=24)
    t.add_column("Value", style="white")

    t.add_row("Cash budget", f"${portfolio['budget']:,.2f}")
    t.add_row("Tickers", f"{len(portfolio['tickers'])} — {ticker_display}")

    if profile["llm_provider"] == _AUTO_PROVIDER:
        t.add_row("LLM provider", "[dim]auto (GPU → Ollama/Qwen, else Gemini)[/dim]")
        t.add_row("Deep model", "[dim]resolved at runtime[/dim]")
        t.add_row("Quick model", "[dim]resolved at runtime[/dim]")
    else:
        t.add_row("LLM provider", profile["llm_provider"])
        t.add_row("Deep model", profile["deep_think_llm"])
        t.add_row("Quick model", profile["quick_think_llm"])

    t.add_row("Max workers", str(profile["max_workers"]))
    t.add_row("Profile path", str(PROFILE_PATH))
    t.add_row("Portfolio config", str(PORTFOLIO_CONFIG_PATH))

    console.print(t)
    console.print()


def _ask_budget() -> float:
    raw = questionary.text(
        "Starting cash budget ($):",
        default="10000",
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Enter a valid number",
    ).ask()
    if raw is None:
        sys.exit(0)
    return float(raw)


def _run_screener(universe_source: str) -> list[str]:
    """Run the pre-screener pipeline and return the ranked ticker list."""
    from scripts.pre_screener import screen

    label = _UNIVERSE_LABELS.get(universe_source, universe_source)
    console.print()
    console.print(f"[dim]Universe: {label}[/dim]")

    top_raw = questionary.text(
        "How many top candidates to keep?",
        default="20",
        validate=lambda v: (v.isdigit() and int(v) >= 1) or "Enter a positive number",
    ).ask()
    if top_raw is None:
        sys.exit(0)
    top_n = int(top_raw)

    if universe_source == "full":
        console.print(
            "[yellow]Scanning the full US market (~9,000 tickers). "
            "This takes a few minutes — emerging stocks included.[/yellow]"
        )

    console.print()
    result = screen(top_n=top_n, universe_source=universe_source)

    if not result.get("success"):
        console.print("[red]Pre-screener failed. Please enter tickers manually.[/red]")
        return []

    tickers = result.get("candidates", [])
    console.print(
        f"\n[green]Screener found {len(tickers)} candidates:[/green] "
        + ", ".join(tickers)
    )
    return tickers


def _ask_tickers() -> list[str]:
    console.print(
        "[dim]Enter comma-separated tickers (e.g. NVDA,AAPL,MSFT) or a universe keyword:\n"
        "  all / full       — full US market via NASDAQ FTP (~9,000 tickers, catches emerging stocks)\n"
        "  sp500            — S&P 500 constituents\n"
        "  nasdaq100        — Nasdaq 100 constituents\n"
        "  sp500+nasdaq100  — union of both[/dim]"
    )
    raw = questionary.text(
        "Tickers / universe:",
        validate=lambda v: len(v.strip()) > 0 or "Enter tickers or a universe keyword",
    ).ask()
    if raw is None:
        sys.exit(0)

    keyword = raw.strip().lower()
    if keyword in _UNIVERSE_KEYWORDS:
        universe_source = _UNIVERSE_KEYWORDS[keyword]
        tickers = _run_screener(universe_source)
        if tickers:
            return tickers
        # Fall through to manual entry on screener failure
        console.print("[yellow]Falling back to manual ticker entry.[/yellow]")
        return _ask_tickers()

    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _ask_provider() -> tuple[str, str, str]:
    from tradingagents.llm_clients.auto_detect import detect_provider_verbose

    provider = questionary.select(
        "LLM provider:",
        choices=[
            questionary.Choice(
                "Auto-detect  [GPU present → Ollama/Qwen, otherwise Gemini]",
                value=_AUTO_PROVIDER,
            ),
            questionary.Choice(
                f"Google Gemini  [{PROVIDER_MODELS['google']['note']}]", value="google"
            ),
            questionary.Choice(
                f"Ollama (local)  [{PROVIDER_MODELS['ollama']['note']}]", value="ollama"
            ),
            questionary.Choice(
                f"OpenAI  [{PROVIDER_MODELS['openai']['note']}]", value="openai"
            ),
            questionary.Choice(
                f"Anthropic  [{PROVIDER_MODELS['anthropic']['note']}]", value="anthropic"
            ),
        ],
    ).ask()
    if provider is None:
        sys.exit(0)

    if provider == _AUTO_PROVIDER:
        cfg, reason = detect_provider_verbose()
        console.print(f"  [dim]Detected: {reason}[/dim]")
        # Store the sentinel; nightly_analysis resolves it at runtime
        return _AUTO_PROVIDER, _AUTO_PROVIDER, _AUTO_PROVIDER

    defaults = PROVIDER_MODELS[provider]

    deep = questionary.text(
        "Deep-think model (for researchers, trader, portfolio manager):",
        default=defaults["deep"],
    ).ask()
    if deep is None:
        sys.exit(0)

    quick = questionary.text(
        "Quick-think model (for data-fetching analysts):",
        default=defaults["quick"],
    ).ask()
    if quick is None:
        sys.exit(0)

    return provider, deep.strip(), quick.strip()


def _ask_workers() -> int:
    raw = questionary.text(
        "Max parallel analysis workers (1–10):",
        default="3",
        validate=lambda v: (v.isdigit() and 1 <= int(v) <= 10) or "Enter a number 1–10",
    ).ask()
    if raw is None:
        sys.exit(0)
    return int(raw)


def _ask_limit_buffer() -> float:
    console.print(
        "[dim]Limit price buffer: buys are placed this % above the estimated price to\n"
        "  ensure fill; sells this % below. 0.5% is a safe default for liquid US stocks.[/dim]"
    )
    raw = questionary.text(
        "Limit price buffer % (e.g. 0.5):",
        default="0.5",
        validate=lambda v: (
            v.replace(".", "", 1).isdigit() and 0.0 <= float(v) <= 5.0
        ) or "Enter a number between 0 and 5",
    ).ask()
    if raw is None:
        sys.exit(0)
    return float(raw)


def _ask_existing_holdings(tickers: list[str]) -> dict[str, float]:
    has_holdings = questionary.confirm(
        "Do you already hold any of these stocks?", default=False
    ).ask()
    if not has_holdings:
        return {}

    holdings = {}
    for ticker in tickers:
        units_raw = questionary.text(
            f"  Units of {ticker} you currently hold (0 to skip):",
            default="0",
            validate=lambda v: (v.replace(".", "", 1).isdigit()) or "Enter a number",
        ).ask()
        if units_raw is None:
            sys.exit(0)
        units = float(units_raw)
        if units > 0:
            holdings[ticker] = units
    return holdings


def run_wizard():
    _print_header()

    # Check if a profile already exists
    if PROFILE_PATH.exists():
        overwrite = questionary.confirm(
            f"Profile already exists at {PROFILE_PATH}. Overwrite?", default=False
        ).ask()
        if not overwrite:
            console.print("[yellow]Wizard cancelled. Existing profile unchanged.[/yellow]")
            return

    console.print("[dim]Step 1 of 5 — Portfolio[/dim]")
    budget = _ask_budget()
    tickers = _ask_tickers()

    console.print()
    console.print("[dim]Step 2 of 5 — Existing holdings[/dim]")
    holdings = _ask_existing_holdings(tickers)

    console.print()
    console.print("[dim]Step 3 of 5 — LLM provider[/dim]")
    provider, deep_model, quick_model = _ask_provider()

    console.print()
    console.print("[dim]Step 4 of 5 — Performance & order settings[/dim]")
    max_workers = _ask_workers()
    limit_buffer = _ask_limit_buffer()

    console.print()
    console.print("[dim]Step 5 of 5 — Confirm[/dim]")

    profile = {
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "llm_provider": provider,
        "deep_think_llm": deep_model,
        "quick_think_llm": quick_model,
        "max_workers": max_workers,
        "limit_price_buffer_pct": limit_buffer,
    }

    portfolio = {
        "tickers": tickers,
        "budget": budget,
        "holdings": holdings,
        "max_workers": max_workers,
    }

    _print_summary(profile, portfolio)

    confirmed = questionary.confirm("Save this profile?", default=True).ask()
    if not confirmed:
        console.print("[yellow]Wizard cancelled.[/yellow]")
        return

    TRADINGAGENTS_HOME.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    PORTFOLIO_CONFIG_PATH.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")

    console.print()
    console.print("[bold green]Profile saved![/bold green]")
    console.print(f"  Profile:   [cyan]{PROFILE_PATH}[/cyan]")
    console.print(f"  Portfolio: [cyan]{PORTFOLIO_CONFIG_PATH}[/cyan]")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print(
        "  Run analysis:    [bold]python scripts/nightly_analysis.py[/bold]"
    )
    console.print(
        "  Update holdings: [bold]python scripts/update_holdings.py[/bold]"
    )
    console.print()


if __name__ == "__main__":
    run_wizard()
