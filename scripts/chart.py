"""Price chart generator for BUY signal announcements.

Generates a 60-day candlestick chart with volume, SMA20/SMA50, and an entry
price marker. Returns the chart as a PNG bytes object ready to send via Telegram.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_buy_chart(
    ticker: str,
    date: str,
    entry_price: Optional[float] = None,
    lookback_days: int = 60,
) -> Optional[bytes]:
    """Generate a candlestick chart for a BUY signal.

    Args:
        ticker: Ticker symbol
        date: Analysis date (YYYY-MM-DD)
        entry_price: Entry price to mark with a horizontal line (optional)
        lookback_days: Number of calendar days of history to show

    Returns:
        PNG image as bytes, or None if data unavailable / chart fails
    """
    try:
        import mplfinance as mpf
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("mplfinance not installed — skipping chart generation")
        return None

    # Fetch OHLCV data from the incremental cache first, fall back to yfinance
    df = _fetch_ohlcv(ticker, date, lookback_days)
    if df is None or len(df) < 10:
        logger.warning("Insufficient OHLCV data for %s chart", ticker)
        return None

    try:
        # Build addplots: SMA20, SMA50
        # Compute on full df, then align to the display window
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        add_plots = [mpf.make_addplot(sma20, color="#2196F3", width=1.2)]
        # Only include SMA50 if we have enough non-NaN values to avoid mpf errors
        if sma50.notna().sum() >= 5:
            add_plots.append(mpf.make_addplot(sma50, color="#FF9800", width=1.2))

        # Horizontal entry price line
        hlines = {}
        if entry_price:
            hlines = dict(hlines=entry_price)

        # Dark style matching Telegram dark mode
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mpf.make_marketcolors(
                up="#26A69A", down="#EF5350",
                edge="inherit",
                wick="inherit",
                volume={"up": "#26A69A55", "down": "#EF535055"},
            ),
            facecolor="#1C1C1E",
            edgecolor="#2C2C2E",
            figcolor="#1C1C1E",
            gridcolor="#2C2C2E",
            gridstyle="--",
            gridaxis="both",
            rc={"font.size": 9, "axes.labelcolor": "#EBEBF5", "xtick.color": "#EBEBF5",
                "ytick.color": "#EBEBF5", "text.color": "#EBEBF5"},
        )

        title = f"{ticker}  ▲ BUY Signal  {date}"
        fig, axes = mpf.plot(
            df,
            type="candle",
            volume=True,
            addplot=add_plots,
            style=style,
            title=title,
            ylabel="Price (USD)",
            ylabel_lower="Volume",
            figsize=(10, 6),
            tight_layout=True,
            returnfig=True,
            **hlines,
        )

        # Legend
        patches = [
            mpatches.Patch(color="#2196F3", label="SMA 20"),
            mpatches.Patch(color="#FF9800", label="SMA 50"),
        ]
        if entry_price:
            patches.append(mpatches.Patch(color="#00E676", label=f"Entry ${entry_price:,.2f}"))
        axes[0].legend(handles=patches, loc="upper left", facecolor="#2C2C2E",
                       edgecolor="#3C3C3E", labelcolor="#EBEBF5", fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="#1C1C1E", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        logger.warning("Chart generation failed for %s: %s", ticker, exc)
        return None


def _fetch_ohlcv(ticker: str, date: str, lookback_days: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from the cache or yfinance, return a mplfinance-ready DataFrame."""
    end_dt = datetime.strptime(date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=int(lookback_days * 1.5))  # wider to ensure enough bars
    start_str = start_dt.strftime("%Y-%m-%d")

    # Try cache first
    try:
        from tradingagents.dataflows.ohlcv_cache import _cache
        df = _cache.get_ohlcv(ticker, start_str, date)
        if df is not None and len(df) >= 10:
            return _normalise(df, lookback_days)
    except Exception:
        pass

    # Fall back to live yfinance
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start_str, end=date,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        return _normalise(raw, lookback_days)
    except Exception as exc:
        logger.warning("yfinance fallback failed for %s: %s", ticker, exc)
        return None


def _normalise(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Ensure DataFrame has OHLCV columns and is trimmed to lookback_days trading days."""
    rename = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            df[col] = df.get("Close", 0)
    df = df[needed].dropna(subset=["Close"])
    # Keep roughly N trading days (lookback_days calendar days ≈ 0.71 trading days)
    # Use at least 60 bars so SMA50 has enough data to render
    trading_days = max(60, int(lookback_days * 0.71))
    return df.tail(trading_days)
