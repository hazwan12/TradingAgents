#!/usr/bin/env python3
"""Sync live holdings from Moomoo OpenD into portfolio_config.json.

Prerequisites
-------------
1. Install the SDK:
       pip install moomoo-openapi
2. Download and launch OpenD (the local gateway) from:
       https://www.moomoo.com/download/openD
   It runs on 127.0.0.1:11111 by default and must be running before this script.

Usage
-----
    python scripts/sync_moomoo_holdings.py              # preview + confirm
    python scripts/sync_moomoo_holdings.py --dry-run    # print positions, no save
    python scripts/sync_moomoo_holdings.py --yes        # skip confirmation prompt
    python scripts/sync_moomoo_holdings.py --host 127.0.0.1 --port 11111
    python scripts/sync_moomoo_holdings.py --paper      # use paper-trade account
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console()

TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
PORTFOLIO_CONFIG_PATH = TRADINGAGENTS_HOME / "portfolio_config.json"

# Maps Moomoo market prefix → yfinance suffix transformation
# US.NVDA  → NVDA      (strip prefix)
# SG.D05   → D05.SI    (strip prefix, add .SI)
_MARKET_MAP = {
    "US": lambda code: code,
    "SG": lambda code: f"{code}.SI",
    "HK": lambda code: f"{code}.HK",
}


def _moomoo_code_to_ticker(moomoo_code: str) -> str:
    """Convert 'US.NVDA' → 'NVDA', 'SG.D05' → 'D05.SI', etc."""
    if "." not in moomoo_code:
        return moomoo_code
    market, code = moomoo_code.split(".", 1)
    transform = _MARKET_MAP.get(market.upper())
    if transform:
        return transform(code)
    return moomoo_code  # unknown market — return as-is


def _fetch_positions(host: str, port: int, paper: bool) -> list[dict]:
    """Connect to OpenD and return list of {ticker, qty, cost_price, market_val}."""
    try:
        import moomoo as ft
    except ImportError:
        console.print(
            "[bold red]moomoo-openapi not installed.[/bold red]\n"
            "Run:  [bold]pip install moomoo-openapi[/bold]"
        )
        sys.exit(1)

    trd_env = ft.TrdEnv.SIMULATE if paper else ft.TrdEnv.REAL

    markets_to_try = [ft.TrdMarket.US, ft.TrdMarket.SG]
    all_positions: dict[str, dict] = {}

    for market in markets_to_try:
        market_name = market.value if hasattr(market, "value") else str(market)
        try:
            ctx = ft.OpenSecTradeContext(filter_trdmarket=market, host=host, port=port)
        except Exception as exc:
            console.print(
                f"[yellow]Could not open trade context for {market_name}: {exc}[/yellow]"
            )
            continue

        try:
            ret, data = ctx.position_list_query(trd_env=trd_env)
            if ret != ft.RET_OK:
                console.print(
                    f"[yellow]position_list_query failed for {market_name}: {data}[/yellow]"
                )
                continue

            if data.empty:
                continue

            for _, row in data.iterrows():
                moomoo_code = str(row.get("code", ""))
                qty = float(row.get("qty", 0))
                if qty <= 0:
                    continue
                ticker = _moomoo_code_to_ticker(moomoo_code)
                all_positions[ticker] = {
                    "ticker": ticker,
                    "moomoo_code": moomoo_code,
                    "qty": qty,
                    "cost_price": float(row.get("cost_price", 0)),
                    "market_val": float(row.get("market_val", 0)),
                    "nominal_price": float(row.get("nominal_price", 0)),
                    "pl_val": float(row.get("pl_val", 0)),
                }
        except Exception as exc:
            console.print(f"[yellow]Error querying {market_name} positions: {exc}[/yellow]")
        finally:
            try:
                ctx.close()
            except Exception:
                pass

    return list(all_positions.values())


def _fetch_cash(host: str, port: int, paper: bool) -> float | None:
    """Return available cash from the US account, or None on failure."""
    try:
        import moomoo as ft
    except ImportError:
        return None

    trd_env = ft.TrdEnv.SIMULATE if paper else ft.TrdEnv.REAL
    try:
        ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=host, port=port)
        ret, data = ctx.accinfo_query(trd_env=trd_env)
        ctx.close()
        if ret == ft.RET_OK and not data.empty:
            return float(data.iloc[0].get("cash", 0))
    except Exception:
        pass
    return None


def _print_positions_table(positions: list[dict]):
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    t.add_column("Ticker", style="bold white")
    t.add_column("Moomoo Code", style="dim")
    t.add_column("Qty", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("Last Price", justify="right")
    t.add_column("Market Val", justify="right")
    t.add_column("P&L", justify="right")

    for p in sorted(positions, key=lambda x: x["ticker"]):
        pl_style = "green" if p["pl_val"] >= 0 else "red"
        t.add_row(
            p["ticker"],
            p["moomoo_code"],
            f"{p['qty']:.4f}",
            f"${p['cost_price']:,.4f}" if p["cost_price"] else "—",
            f"${p['nominal_price']:,.4f}" if p["nominal_price"] else "—",
            f"${p['market_val']:,.2f}" if p["market_val"] else "—",
            f"[{pl_style}]${p['pl_val']:+,.2f}[/{pl_style}]" if p["pl_val"] else "—",
        )

    console.print(t)


def _load_portfolio() -> dict:
    if PORTFOLIO_CONFIG_PATH.exists():
        return json.loads(PORTFOLIO_CONFIG_PATH.read_text())
    return {"tickers": [], "budget": 0.0, "holdings": {}}


def _save_portfolio(portfolio: dict):
    PORTFOLIO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_CONFIG_PATH.write_text(json.dumps(portfolio, indent=2))


def run(host: str, port: int, dry_run: bool, yes: bool, paper: bool):
    env_label = "[yellow](PAPER)[/yellow]" if paper else "[green](LIVE)[/green]"
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]TradingAgents — Moomoo Holdings Sync[/bold cyan]\n"
            f"[dim]OpenD  {host}:{port}  {env_label}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    console.print(f"[dim]Connecting to OpenD at {host}:{port} …[/dim]")
    positions = _fetch_positions(host, port, paper)

    if not positions:
        console.print("[yellow]No positions returned from Moomoo.[/yellow]")
        return

    console.print(f"Found [bold]{len(positions)}[/bold] position(s):\n")
    _print_positions_table(positions)

    # Fetch cash
    cash = _fetch_cash(host, port, paper)
    if cash is not None:
        console.print(f"\n  Cash (US account): [bold]${cash:,.2f}[/bold]")

    if dry_run:
        console.print("\n[dim]Dry run — no changes written.[/dim]")
        return

    console.print()
    console.print(Rule("[dim]Save to portfolio_config.json[/dim]"))
    console.print(
        f"[dim]This will [bold]replace[/bold] all holdings in "
        f"[cyan]{PORTFOLIO_CONFIG_PATH}[/cyan][/dim]"
    )

    if not yes:
        try:
            import questionary
            confirmed = questionary.confirm("Save these holdings?", default=True).ask()
            if not confirmed:
                console.print("[yellow]Cancelled.[/yellow]")
                return
        except ImportError:
            # questionary unavailable — require --yes
            console.print(
                "[yellow]questionary not available. Re-run with [bold]--yes[/bold] to skip prompt.[/yellow]"
            )
            return

    portfolio = _load_portfolio()

    # Replace holdings
    new_holdings = {p["ticker"]: p["qty"] for p in positions}
    portfolio["holdings"] = new_holdings

    # Sync tickers list to match held positions (add any new; keep existing watchlist entries too)
    held_tickers = set(new_holdings.keys())
    existing_tickers = set(portfolio.get("tickers", []))
    portfolio["tickers"] = sorted(existing_tickers | held_tickers)

    # Update cash if fetched
    if cash is not None:
        portfolio["budget"] = round(cash, 2)

    portfolio["updated_at"] = datetime.now().isoformat()
    portfolio["moomoo_synced_at"] = datetime.now().isoformat()

    _save_portfolio(portfolio)

    console.print(f"\n[bold green]Saved.[/bold green]  {PORTFOLIO_CONFIG_PATH}")
    console.print(f"  {len(new_holdings)} position(s) written.")
    if cash is not None:
        console.print(f"  Cash updated: ${cash:,.2f}")
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Sync live Moomoo holdings into portfolio_config.json"
    )
    parser.add_argument("--host", default="127.0.0.1", help="OpenD host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11111, help="OpenD port (default: 11111)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print positions without saving",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use paper-trade (simulated) account instead of live",
    )
    args = parser.parse_args()
    run(host=args.host, port=args.port, dry_run=args.dry_run, yes=args.yes, paper=args.paper)


if __name__ == "__main__":
    main()
