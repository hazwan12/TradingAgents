#!/usr/bin/env python3
"""Update holdings after trades execute — marks orders done and rebalances portfolio_config.json.

Usage:
    python scripts/update_holdings.py                  # interactive: pick from pending order files
    python scripts/update_holdings.py --file ORDERS_2026-06-02.json   # specify file directly
    python scripts/update_holdings.py --list           # list pending order files

For each pending order you confirm:
  - Whether it was executed (y/n)
  - Actual execution price (defaults to estimated)
  - Actual units executed (defaults to staged units)

Holdings and budget in portfolio_config.json are updated and an audit entry is appended
to ~/.tradingagents/transactions.jsonl.
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
from rich.table import Table
from rich import box
import questionary

load_dotenv()

console = Console()

TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
PORTFOLIO_CONFIG_PATH = TRADINGAGENTS_HOME / "portfolio_config.json"
ORDERS_DIR = TRADINGAGENTS_HOME / "orders"
TRANSACTIONS_PATH = TRADINGAGENTS_HOME / "transactions.jsonl"


def _load_portfolio() -> dict:
    if not PORTFOLIO_CONFIG_PATH.exists():
        console.print(
            "[bold red]No portfolio config found.[/bold red] "
            "Run [bold]python scripts/create_profile.py[/bold] first."
        )
        sys.exit(1)
    return json.loads(PORTFOLIO_CONFIG_PATH.read_text())


def _save_portfolio(portfolio: dict):
    PORTFOLIO_CONFIG_PATH.write_text(json.dumps(portfolio, indent=2))


def _load_orders(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_orders(orders_doc: dict, path: Path):
    path.write_text(json.dumps(orders_doc, indent=2))


def _append_transaction(record: dict):
    TRANSACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRANSACTIONS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _list_order_files() -> list[Path]:
    if not ORDERS_DIR.exists():
        return []
    files = sorted(ORDERS_DIR.glob("ORDERS_*.json"), reverse=True)
    return [f for f in files if f.is_file()]


def _pending_orders_in(orders_doc: dict) -> list[dict]:
    return [o for o in orders_doc.get("orders", []) if o["status"] == "pending"]


def _print_order_summary(order: dict):
    direction_style = "green" if order["direction"] == "buy" else "red"
    limit_str = (
        f"  limit [bold]${order['limit_price']:,.4f}[/bold]"
        if "limit_price" in order
        else ""
    )
    console.print(
        f"\n  [bold]{order['ticker']}[/bold] "
        f"[{direction_style}]{order['direction'].upper()}[/{direction_style}] "
        f"{order['units']:.2f} units @ est. ${order['estimated_price']:,.2f}"
        f"{limit_str}  "
        f"(~${order['estimated_total']:,.2f})  "
        f"Rating: {order.get('rating', '—')}"
    )


def _process_order(order: dict, portfolio: dict) -> bool:
    """Walk user through confirming/cancelling a single order. Returns True if modified."""
    _print_order_summary(order)

    executed = questionary.confirm("Was this order executed?", default=True).ask()
    if executed is None:
        sys.exit(0)

    if not executed:
        cancel = questionary.confirm("Mark as cancelled?", default=True).ask()
        if cancel is None:
            sys.exit(0)
        order["status"] = "cancelled"
        order["executed_at"] = datetime.now().isoformat()
        return True

    # Actual execution details
    price_raw = questionary.text(
        f"  Execution price (est. ${order['estimated_price']:,.2f}):",
        default=str(order["estimated_price"]),
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Enter a valid number",
    ).ask()
    if price_raw is None:
        sys.exit(0)
    exec_price = float(price_raw)

    units_raw = questionary.text(
        f"  Units executed (staged: {order['units']:.2f}):",
        default=str(order["units"]),
        validate=lambda v: v.replace(".", "", 1).isdigit() or "Enter a valid number",
    ).ask()
    if units_raw is None:
        sys.exit(0)
    exec_units = float(units_raw)

    notes = questionary.text("  Notes (optional):", default="").ask() or ""

    # Update order record
    order["status"] = "executed"
    order["executed_at"] = datetime.now().isoformat()
    order["executed_price"] = exec_price
    order["executed_units"] = exec_units
    order["notes"] = notes

    # Update holdings
    ticker = order["ticker"]
    current_units = portfolio["holdings"].get(ticker, 0.0)

    if order["direction"] == "buy":
        new_units = current_units + exec_units
        cost = exec_price * exec_units
        portfolio["budget"] = max(0.0, portfolio.get("budget", 0.0) - cost)
    else:
        new_units = max(0.0, current_units - exec_units)
        proceeds = exec_price * exec_units
        portfolio["budget"] = portfolio.get("budget", 0.0) + proceeds

    if new_units > 0:
        portfolio["holdings"][ticker] = round(new_units, 6)
    elif ticker in portfolio["holdings"]:
        del portfolio["holdings"][ticker]

    total_value = exec_price * exec_units

    # Append audit record
    _append_transaction(
        {
            "timestamp": order["executed_at"],
            "order_id": order["id"],
            "ticker": ticker,
            "direction": order["direction"],
            "units": exec_units,
            "price": exec_price,
            "total": round(total_value, 2),
            "notes": notes,
        }
    )

    console.print(
        f"  [green]Updated[/green] {ticker}: {current_units:.2f} → {new_units:.2f} units"
    )
    return True


def _print_holdings_table(portfolio: dict):
    holdings = portfolio.get("holdings", {})
    console.print()
    console.print(Rule("[bold green]Updated Portfolio[/bold green]"))
    console.print(f"  Cash available: [bold]${portfolio.get('budget', 0.0):,.2f}[/bold]")
    console.print()

    if not holdings:
        console.print("  [dim]No positions held.[/dim]")
        return

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    t.add_column("Ticker", style="bold white")
    t.add_column("Units", justify="right")

    for ticker, units in sorted(holdings.items()):
        t.add_row(ticker, f"{units:.4f}")

    console.print(t)


def run(orders_file: str | None = None, list_only: bool = False):
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Update Holdings[/bold cyan]\n"
            "[dim]Records executed trades and updates your portfolio[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    order_files = _list_order_files()

    if list_only:
        if not order_files:
            console.print("[yellow]No order files found.[/yellow]")
            return
        console.print("[bold]Pending order files:[/bold]")
        for f in order_files:
            doc = _load_orders(f)
            pending = len(_pending_orders_in(doc))
            total = len(doc.get("orders", []))
            console.print(f"  {f.name}  ({pending} pending / {total} total)")
        return

    # Resolve which file to process
    if orders_file:
        path = ORDERS_DIR / orders_file if not Path(orders_file).is_absolute() else Path(orders_file)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
    else:
        if not order_files:
            console.print(
                "[yellow]No order files found.[/yellow] "
                "Run [bold]python scripts/nightly_analysis.py[/bold] first."
            )
            return

        choices = [f.name for f in order_files]
        selected = questionary.select(
            "Which order file to process?",
            choices=choices,
        ).ask()
        if selected is None:
            sys.exit(0)
        path = ORDERS_DIR / selected

    orders_doc = _load_orders(path)
    pending = _pending_orders_in(orders_doc)

    if not pending:
        console.print("[yellow]No pending orders in this file.[/yellow]")
        return

    console.print(f"Found [bold]{len(pending)}[/bold] pending orders in [cyan]{path.name}[/cyan]")
    portfolio = _load_portfolio()
    modified = False

    for order in pending:
        changed = _process_order(order, portfolio)
        if changed:
            modified = True
        console.print()

    if modified:
        _save_orders(orders_doc, path)
        portfolio["updated_at"] = datetime.now().isoformat()
        _save_portfolio(portfolio)
        _print_holdings_table(portfolio)
        console.print(f"  Portfolio saved: [cyan]{PORTFOLIO_CONFIG_PATH}[/cyan]")
        console.print(f"  Transaction log: [cyan]{TRANSACTIONS_PATH}[/cyan]")
    else:
        console.print("[dim]No changes made.[/dim]")

    console.print()


def main():
    parser = argparse.ArgumentParser(description="Update holdings after trades execute")
    parser.add_argument(
        "--file",
        metavar="ORDERS_*.json",
        help="Specific orders file to process (default: interactive selection)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pending order files and exit",
    )
    args = parser.parse_args()
    run(orders_file=args.file, list_only=args.list)


if __name__ == "__main__":
    main()
