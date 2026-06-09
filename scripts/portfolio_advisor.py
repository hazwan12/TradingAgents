#!/usr/bin/env python3
"""Portfolio advisor — converts watchlist analysis into specific buy/sell recommendations."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scan_watchlist import WatchlistScanner
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import (
    calculate_units_to_trade,
    estimate_risk_level,
    aggregate_portfolio_value,
    calculate_portfolio_weights,
    get_current_prices,
    parse_position_sizing,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class PortfolioAdvisor:
    """Convert analysis ratings into concrete portfolio rebalancing recommendations."""

    def __init__(
        self,
        budget: float,
        holdings: Dict[str, float],
        max_workers: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize advisor with portfolio state.

        Args:
            budget: Available cash to deploy
            holdings: Dict of ticker -> units owned (e.g., {"NVDA": 10, "AAPL": 5})
            max_workers: Max parallel workers for WatchlistScanner (default 5)
            config: Custom DEFAULT_CONFIG overrides (optional)
        """
        self.budget = budget
        self.holdings = holdings
        self.config = config or DEFAULT_CONFIG.copy()
        self.scanner = WatchlistScanner(max_workers=max_workers, config=self.config)
        self.results_dir = Path(self.config.get("results_dir", "reports"))
        self.results_dir.mkdir(exist_ok=True)

    def advise(self, tickers: List[str], date: str) -> Dict[str, Any]:
        """Generate buy/sell recommendations for tickers.

        Args:
            tickers: List of tickers to analyze (e.g., ["NVDA", "AAPL"])
            date: Analysis date in YYYY-MM-DD format

        Returns:
            Dict with keys:
            - success: bool
            - portfolio_value: total portfolio value
            - cash_available: remaining cash
            - recommendations: list of per-ticker rebalance recommendations
            - summary_csv: path to CSV file
            - summary_md: path to markdown file
            - total_buy_power_needed: $ needed to execute buys
            - total_sell_proceeds: $ from executing sells
        """
        logger.info(f"Generating portfolio advice for {len(tickers)} tickers on {date}")

        # Step 1: Run watchlist scanner
        logger.info("Step 1: Running watchlist analysis...")
        scan_result = self.scanner.scan(tickers, date)
        if not scan_result.get("results"):
            logger.error("Watchlist scan failed or returned no results")
            return {"success": False, "error": "Watchlist scan failed"}

        # Step 2: Fetch current prices
        logger.info("Step 2: Fetching current prices...")
        prices = get_current_prices(tickers)
        if not prices:
            logger.error("Failed to fetch any prices")
            return {"success": False, "error": "Failed to fetch prices"}

        # Step 3: Calculate portfolio value
        logger.info("Step 3: Calculating portfolio state...")
        total_portfolio_value = aggregate_portfolio_value(
            self.holdings, prices, self.budget
        )
        current_weights = calculate_portfolio_weights(
            self.holdings, prices, self.budget
        )

        # Step 4: Build recommendations
        logger.info("Step 4: Building recommendations...")
        recommendations = []
        total_buy_power = 0.0
        total_sell_proceeds = 0.0

        for scan_result_item in scan_result["results"]:
            ticker = scan_result_item["ticker"]
            action = scan_result_item["action"]

            current_price = prices.get(ticker, 0.0)
            if current_price <= 0:
                logger.warning(f"Skipping {ticker}: no price data")
                continue

            current_units = self.holdings.get(ticker, 0.0)

            # Parse position sizing from scan result
            position_sizing_text = scan_result_item.get("executive_summary", "")
            target_pct = parse_position_sizing(position_sizing_text)

            # Calculate units to trade
            units_to_trade, dollar_amount = calculate_units_to_trade(
                current_price, target_pct, total_portfolio_value, current_units
            )

            # Estimate risk
            risk_level = estimate_risk_level(action, current_units, units_to_trade)

            # Calculate new weights after trade
            projected_units = current_units + units_to_trade
            projected_value = projected_units * current_price
            projected_weight = (
                (projected_value / total_portfolio_value) * 100.0
                if total_portfolio_value > 0
                else 0.0
            )

            current_weight = current_weights.get(ticker, 0.0)

            recommendation = {
                "ticker": ticker,
                "action": action,
                "rating": scan_result_item.get("rating", "Hold"),
                "current_price": round(current_price, 2),
                "units_held": round(current_units, 2),
                "units_to_trade": round(units_to_trade, 2),
                "dollar_amount": round(dollar_amount, 2),
                "target_allocation_pct": round(target_pct * 100, 2),
                "current_allocation_pct": round(current_weight, 2),
                "projected_allocation_pct": round(projected_weight, 2),
                "risk_level": risk_level,
            }

            recommendations.append(recommendation)

            # Track buy/sell power
            if units_to_trade > 0:
                total_buy_power += dollar_amount
            else:
                total_sell_proceeds += dollar_amount

        # Step 5: Generate outputs
        logger.info("Step 5: Generating output files...")
        summary_csv = self._write_csv_summary(recommendations, date)
        summary_md = self._write_markdown_summary(
            recommendations, total_portfolio_value, self.budget, date
        )

        result = {
            "success": True,
            "portfolio_value": round(total_portfolio_value, 2),
            "cash_available": round(self.budget, 2),
            "recommendations": recommendations,
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "total_buy_power_needed": round(total_buy_power, 2),
            "total_sell_proceeds": round(total_sell_proceeds, 2),
            "net_rebalance": round(total_sell_proceeds - total_buy_power, 2),
        }

        logger.info(
            f"Portfolio advice generated: {len(recommendations)} tickers, "
            f"${total_buy_power:.2f} to buy, ${total_sell_proceeds:.2f} to sell"
        )

        return result

    def _write_csv_summary(self, recommendations: List[Dict], date: str) -> Path:
        """Write CSV summary of recommendations.

        Args:
            recommendations: List of per-ticker recommendations
            date: Analysis date

        Returns:
            Path to CSV file
        """
        df = pd.DataFrame(recommendations)
        csv_path = self.results_dir / f"ADVISOR_{date}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV summary written to {csv_path}")
        return csv_path

    def _write_markdown_summary(
        self,
        recommendations: List[Dict],
        portfolio_value: float,
        budget: float,
        date: str,
    ) -> Path:
        """Write markdown summary of recommendations.

        Args:
            recommendations: List of per-ticker recommendations
            portfolio_value: Total portfolio value
            budget: Available cash
            date: Analysis date

        Returns:
            Path to markdown file
        """
        md_path = self.results_dir / f"ADVISOR_{date}.md"

        lines = [
            "# Portfolio Rebalancing Advisor",
            "",
            f"**Date**: {date}",
            f"**Portfolio Value**: ${portfolio_value:,.2f}",
            f"**Available Cash**: ${budget:,.2f}",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "## Recommendations",
            "",
            "| Ticker | Action | Rating | Price | Units Held | Units to Trade | $ Amount | Current % | Target % | Risk |",
            "|--------|--------|--------|-------|-----------|-----------------|----------|-----------|----------|------|",
        ]

        for rec in sorted(recommendations, key=lambda x: x["ticker"]):
            units_str = f"{rec['units_to_trade']:+.2f}" if rec["units_to_trade"] != 0 else "—"
            amount_str = f"${rec['dollar_amount']:,.2f}" if rec["dollar_amount"] > 0 else "—"
            lines.append(
                f"| {rec['ticker']} | {rec['action']} | {rec['rating']} | "
                f"${rec['current_price']:.2f} | {rec['units_held']:.2f} | {units_str} | {amount_str} | "
                f"{rec['current_allocation_pct']:.1f}% | {rec['target_allocation_pct']:.1f}% | {rec['risk_level']} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## Legend",
                "",
                "- **Units to Trade**: Positive = buy, Negative = sell (number of shares)",
                "- **Current %**: Current allocation as % of portfolio",
                "- **Target %**: Recommended allocation based on analysis",
                "- **Risk**: Low/Medium/High based on action and position change",
                "",
            ]
        )

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown summary written to {md_path}")
        return md_path


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/portfolio_advisor.py <config.json>",
            file=sys.stderr,
        )
        print(
            "  config.json should contain: tickers, date, budget, holdings (optional: max_workers)",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("Example config.json:", file=sys.stderr)
        print(
            json.dumps(
                {
                    "tickers": ["NVDA", "AAPL"],
                    "date": "2026-06-02",
                    "budget": 10000,
                    "holdings": {"NVDA": 10},
                    "max_workers": 5,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    load_dotenv()

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract required fields
    tickers = config_data.get("tickers", [])
    date = config_data.get("date")
    budget = config_data.get("budget", 0)
    holdings = config_data.get("holdings", {})
    max_workers = config_data.get("max_workers", 5)

    if not tickers or not date or budget <= 0:
        print(
            "Error: Config must have tickers, date, and budget > 0",
            file=sys.stderr,
        )
        sys.exit(1)

    # Clean up holdings (ensure floats)
    holdings = {k: float(v) for k, v in holdings.items()}

    trading_config = DEFAULT_CONFIG.copy()
    advisor = PortfolioAdvisor(
        budget=budget, holdings=holdings, max_workers=max_workers, config=trading_config
    )

    try:
        result = advisor.advise(tickers, date)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)
    except Exception as e:
        logger.exception("Fatal error during portfolio advice")
        print(
            json.dumps({"success": False, "error": str(e)}, indent=2),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
