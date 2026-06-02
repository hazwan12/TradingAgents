#!/usr/bin/env python3
"""Watchlist scanner — analyzes multiple tickers in parallel and produces a summary table."""

import re
import sys
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.rating import parse_rating


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class WatchlistScanner:
    """Parallel watchlist analyzer using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 5, config: Optional[Dict[str, Any]] = None):
        """Initialize scanner with optional custom config.

        Args:
            max_workers: Maximum concurrent ticker analyses (default 5)
            config: Custom DEFAULT_CONFIG overrides (optional)
        """
        self.max_workers = max_workers
        self.config = config or DEFAULT_CONFIG.copy()
        self.results_dir = Path(self.config.get("results_dir", "reports"))
        self.results_dir.mkdir(exist_ok=True)

    def scan(self, tickers: List[str], date: str) -> Dict[str, Any]:
        """Scan multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols (e.g., ["NVDA", "AAPL"])
            date: Analysis date in YYYY-MM-DD format

        Returns:
            Dict with keys:
            - results: List of analysis results (one per ticker)
            - failed: List of (ticker, error) tuples
            - summary_csv: Path to CSV summary
            - summary_md: Path to markdown summary
        """
        logger.info(f"Starting scan of {len(tickers)} tickers on {date}")
        start_time = datetime.now()

        results = []
        failed = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs and map ticker to future
            futures = {
                executor.submit(self._analyze_ticker, ticker, date): ticker
                for ticker in tickers
            }

            # Process completed jobs as they finish
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✓ {ticker}: {result['rating']} ({result['action']})")
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    failed.append((ticker, error_msg))
                    logger.error(f"✗ {ticker}: {error_msg}")

        elapsed = datetime.now() - start_time

        # Generate summary outputs
        summary_csv = self._write_csv_summary(results, date)
        summary_md = self._write_markdown_summary(results, failed, date, elapsed)

        logger.info(
            f"Scan complete: {len(results)}/{len(tickers)} succeeded in {elapsed.total_seconds():.1f}s"
        )

        return {
            "results": results,
            "failed": failed,
            "summary_csv": summary_csv,
            "summary_md": summary_md,
            "elapsed_seconds": elapsed.total_seconds(),
        }

    def _analyze_ticker(self, ticker: str, date: str) -> Dict[str, Any]:
        """Analyze a single ticker.

        Args:
            ticker: Ticker symbol
            date: Analysis date

        Returns:
            Dict with extracted decision fields: ticker, rating, action, entry_price, etc.

        Raises:
            Any exception from TradingAgentsGraph.propagate()
        """
        logger.info(f"Analyzing {ticker}...")

        # Create graph instance for this ticker (thread-safe)
        graph = TradingAgentsGraph(config=self.config)

        # Run analysis
        final_state, _ = graph.propagate(ticker, date)

        # Extract key fields from final_state
        rating_text = final_state.get("final_trade_decision", "")
        rating = parse_rating(rating_text)

        trader_text = final_state.get("trader_investment_decision", "")
        action = self._extract_action(trader_text)
        entry_price = self._extract_price(trader_text, "Entry Price")
        stop_loss = self._extract_price(trader_text, "Stop Loss")

        portfolio_text = final_state.get("final_trade_decision", "")
        price_target = self._extract_price(portfolio_text, "Price Target")

        # Extract executive summary from portfolio decision
        summary = self._extract_section(portfolio_text, "Executive Summary")

        return {
            "ticker": ticker,
            "rating": rating.value if hasattr(rating, "value") else str(rating),
            "action": action,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "price_target": price_target,
            "executive_summary": summary,
            "date": date,
        }

    def _extract_action(self, text: str) -> str:
        """Extract trader action (Buy/Hold/Sell) from markdown."""
        match = re.search(r"\*\*Action\*\*:\s*(\w+)", text)
        if match:
            return match.group(1)
        # Fallback: check for FINAL TRANSACTION PROPOSAL
        match = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*\*(\w+)\*\*", text)
        return match.group(1) if match else "Hold"

    def _extract_price(self, text: str, label: str) -> Optional[float]:
        """Extract a numeric price value from markdown (e.g., Entry Price: 225.00)."""
        pattern = rf"\*\*{label}\*\*:\s*([\d.]+)"
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def _extract_section(self, text: str, section: str, max_chars: int = 150) -> str:
        """Extract and truncate a markdown section (e.g., Executive Summary).

        Args:
            text: Full markdown text
            section: Section header to find
            max_chars: Maximum characters to keep (with ellipsis if truncated)

        Returns:
            Extracted text (truncated if needed)
        """
        # Look for header followed by content until next ## header or EOF
        pattern = rf"\*\*{section}\*\*:\s*(.+?)(?=\n\*\*|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Remove markdown formatting
            content = re.sub(r"[\*_]{1,2}", "", content)
            # Truncate with ellipsis
            if len(content) > max_chars:
                content = content[: max_chars - 3] + "..."
            return content
        return ""

    def _write_csv_summary(self, results: List[Dict], date: str) -> Path:
        """Write results to CSV file.

        Args:
            results: List of analysis result dicts
            date: Analysis date (used in filename)

        Returns:
            Path to CSV file
        """
        df = pd.DataFrame(results)
        csv_path = self.results_dir / f"WATCHLIST_{date}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV summary written to {csv_path}")
        return csv_path

    def _write_markdown_summary(
        self, results: List[Dict], failed: List, date: str, elapsed: Any
    ) -> Path:
        """Write markdown summary with table and metadata.

        Args:
            results: Successful analysis results
            failed: Failed tickers and errors
            date: Analysis date
            elapsed: Timedelta object

        Returns:
            Path to markdown file
        """
        md_path = self.results_dir / f"WATCHLIST_{date}.md"

        lines = [
            "# Watchlist Scan Summary",
            "",
            f"**Date**: {date}",
            f"**Scan Time**: {elapsed.total_seconds():.1f}s",
            f"**Model**: {self.config.get('llm_provider', 'unknown')} "
            f"({self.config.get('deep_think_llm', 'unknown')})",
            f"**Successful**: {len(results)}/{len(results) + len(failed)}",
            "",
        ]

        if failed:
            lines.extend(
                [
                    "## ⚠️ Failed Tickers",
                    "",
                ]
            )
            for ticker, error in failed:
                lines.append(f"- **{ticker}**: {error}")
            lines.append("")

        # Main results table
        lines.extend(
            [
                "## Results",
                "",
                "| Ticker | Rating | Action | Entry Price | Stop Loss | Price Target | Summary |",
                "|--------|--------|--------|-------------|-----------|--------------|---------|",
            ]
        )

        for result in sorted(results, key=lambda x: x["ticker"]):
            entry = result["entry_price"] or "—"
            stop = result["stop_loss"] or "—"
            target = result["price_target"] or "—"
            summary = result["executive_summary"][:60] + "..." if len(result["executive_summary"]) > 60 else result["executive_summary"]
            lines.append(
                f"| {result['ticker']} | {result['rating']} | {result['action']} | {entry} | {stop} | {target} | {summary} |"
            )

        lines.extend(
            [
                "",
                "---",
                f"*Generated {datetime.now().isoformat()}*",
            ]
        )

        md_path.write_text("\n".join(lines))
        logger.info(f"Markdown summary written to {md_path}")
        return md_path


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/scan_watchlist.py <tickers> <date> [max_workers]",
            file=sys.stderr,
        )
        print(
            "  tickers: Comma-separated list (e.g., 'NVDA,AAPL,MSFT')",
            file=sys.stderr,
        )
        print("  date: Analysis date (YYYY-MM-DD)", file=sys.stderr)
        print("  max_workers: Max parallel workers (default 5)", file=sys.stderr)
        sys.exit(1)

    load_dotenv()

    tickers = [t.strip().upper() for t in sys.argv[1].split(",")]
    date = sys.argv[2]
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    config = DEFAULT_CONFIG.copy()
    scanner = WatchlistScanner(max_workers=max_workers, config=config)

    try:
        result = scanner.scan(tickers, date)

        output = {
            "success": len(result["failed"]) == 0,
            "tickers_scanned": len(tickers),
            "tickers_succeeded": len(result["results"]),
            "tickers_failed": len(result["failed"]),
            "elapsed_seconds": result["elapsed_seconds"],
            "summary_csv": str(result["summary_csv"]),
            "summary_md": str(result["summary_md"]),
            "failed_tickers": [
                {"ticker": ticker, "error": error} for ticker, error in result["failed"]
            ],
        }

        print(json.dumps(output, indent=2))
        sys.exit(0 if output["success"] else 1)

    except Exception as e:
        logger.exception("Fatal error during scan")
        print(
            json.dumps(
                {"success": False, "error": str(e)},
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
