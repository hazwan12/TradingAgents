"""Price fetching utilities for portfolio calculations."""

import logging
from typing import Dict, List

import yfinance as yf

from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.dataflows.stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Batch fetch current prices for multiple tickers.

    Uses yfinance with rate limiting via yf_retry. Handles symbol normalization.
    Skips tickers with fetch errors and logs warnings.

    Args:
        tickers: List of ticker symbols (e.g., ["NVDA", "AAPL"])

    Returns:
        Dict of ticker -> current price (e.g., {"NVDA": 189.50, "AAPL": 195.20})
        Excludes tickers that failed to fetch.
    """
    prices = {}

    for ticker in tickers:
        try:
            # Normalize symbol (e.g., BRK.B stays BRK.B, EURUSD → EURUSD=X)
            normalized = normalize_symbol(ticker)

            # Fetch with retry logic
            ticker_obj = yf.Ticker(normalized)
            info = ticker_obj.info

            # Extract current price
            price = info.get("currentPrice")
            if price is None:
                # Fallback to regularMarketPrice
                price = info.get("regularMarketPrice")

            if price and price > 0:
                prices[ticker] = float(price)
                logger.info(f"✓ {ticker}: ${price:.2f}")
            else:
                logger.warning(f"✗ {ticker}: No price data available")

        except Exception as e:
            logger.warning(f"✗ {ticker}: Failed to fetch price — {type(e).__name__}: {str(e)}")

    return prices
