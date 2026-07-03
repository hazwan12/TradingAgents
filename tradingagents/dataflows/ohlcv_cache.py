"""Incremental SQLite cache for yfinance OHLCV data.

Serves three purposes:
  1. Speed — subsequent runs only fetch bars newer than the last cached date
     (1–2 new trading days) instead of re-downloading full lookback windows.
  2. Resilience — if yfinance returns a 401/crumb error, the screener and LLM
     analysts can serve from cache rather than aborting the pipeline.
  3. Backtesting foundation — the database accumulates real historical bars
     passively with every daily run.

Cache location: ~/.tradingagents/cache/ohlcv.db  (SQLite, no extra deps)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta  # noqa: F401 (timedelta used in ensure_cached)
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".tradingagents" / "cache" / "ohlcv.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker  TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL    NOT NULL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON ohlcv (ticker, date);
"""

_DOWNLOAD_BATCH_SIZE = 200


class OHLCVCache:
    """Incremental OHLCV cache backed by a local SQLite database."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cached_date_ranges(self, tickers: List[str]) -> Dict[str, tuple]:
        """Return {ticker: (min_date, max_date)} for each ticker. (None, None) if not cached."""
        placeholders = ",".join("?" * len(tickers))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT ticker, MIN(date), MAX(date) FROM ohlcv "
                f"WHERE ticker IN ({placeholders}) GROUP BY ticker",
                tickers,
            ).fetchall()
        result = {t: (None, None) for t in tickers}
        for ticker, min_date, max_date in rows:
            result[ticker] = (min_date, max_date)
        return result

    def _upsert(self, conn: sqlite3.Connection, records: List[tuple]):
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            records,
        )

    def _fetch_from_yfinance(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Batch-download from yfinance. Returns {ticker: df} with columns
        Open/High/Low/Close/Volume, date-indexed (no tz).

        Uses the same MultiIndex access pattern as the original _batch_download
        in pre_screener.py: raw["Close"] / raw["Volume"] at level 0 = metric.
        """
        results: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), _DOWNLOAD_BATCH_SIZE):
            batch = tickers[i: i + _DOWNLOAD_BATCH_SIZE]
            try:
                raw = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if raw.empty:
                    continue
                if raw.index.tz is not None:
                    raw.index = raw.index.tz_localize(None)

                if isinstance(raw.columns, pd.MultiIndex):
                    # MultiIndex: level 0 = metric (Close/Volume/...), level 1 = ticker
                    # Mirror the original pre_screener.py approach exactly.
                    close_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
                    vol_df = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
                    if isinstance(close_df, pd.Series):
                        # Single-ticker edge case inside multi-ticker download
                        close_df = close_df.to_frame(name=batch[0])
                    if isinstance(vol_df, pd.Series):
                        vol_df = vol_df.to_frame(name=batch[0])
                    for ticker_col in close_df.columns:
                        close_s = close_df[ticker_col].dropna()
                        vol_s = vol_df[ticker_col].dropna() if ticker_col in vol_df.columns else pd.Series(dtype=float)
                        if close_s.empty:
                            continue
                        df = pd.DataFrame({"Open": raw.get("Open", pd.DataFrame()).get(ticker_col),
                                           "High": raw.get("High", pd.DataFrame()).get(ticker_col),
                                           "Low": raw.get("Low", pd.DataFrame()).get(ticker_col),
                                           "Close": close_s,
                                           "Volume": vol_s})
                        df = df.dropna(subset=["Close"])
                        if not df.empty:
                            results[str(ticker_col)] = df
                else:
                    # Single ticker — columns are metric names directly
                    if len(batch) == 1:
                        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
                        df = raw[cols].dropna(subset=["Close"])
                        if not df.empty:
                            results[batch[0]] = df
            except Exception as exc:
                logger.warning("yfinance batch download failed (%s…): %s", batch[0], exc)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_cached(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        batch_size: int = _DOWNLOAD_BATCH_SIZE,
    ):
        """Ensure [start_date, end_date] bars are in the DB for all tickers.

        For tickers already partially cached, only fetches bars newer than the
        last cached date. Falls back gracefully on yfinance failures.
        """
        tickers = list(tickers)
        if not tickers:
            return

        date_ranges = self._cached_date_ranges(tickers)

        # Group tickers by what needs fetching
        cold: List[str] = []         # never cached OR cached range doesn't cover start_date
        warm: Dict[str, str] = {}    # cached tail is recent but end is stale

        for ticker, (min_date, max_date) in date_ranges.items():
            if min_date is None:
                # Never cached
                cold.append(ticker)
            elif min_date > start_date:
                # Cached but doesn't go back far enough — need historical backfill.
                # Fetch the full window so we don't miss early bars needed for RSI/MA.
                cold.append(ticker)
            elif max_date < end_date:
                # Cached far enough back, just need recent tail
                next_day = (
                    datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                warm[ticker] = next_day
            # else: fully cached — nothing to do

        # yfinance's end parameter is EXCLUSIVE — add 1 day to make our end_date inclusive
        yf_end = (
            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        # Cold start — full window
        if cold:
            logger.info("Cache cold-start: fetching %d tickers (%s → %s)", len(cold), start_date, end_date)
            fetched = self._fetch_from_yfinance(cold, start_date, yf_end)
            self._store(fetched)

        # Warm update — tail only
        if warm:
            by_start: Dict[str, List[str]] = {}
            for ticker, s in warm.items():
                by_start.setdefault(s, []).append(ticker)

            for s, batch in by_start.items():
                # Skip if the tail would be a zero-length range (already up to date)
                if s >= yf_end:
                    logger.debug("Cache already up to date for %d tickers", len(batch))
                    continue
                logger.info("Cache update: fetching %d tickers (%s → %s)", len(batch), s, end_date)
                fetched = self._fetch_from_yfinance(batch, s, yf_end)
                self._store(fetched)

    def _store(self, data: Dict[str, pd.DataFrame]):
        if not data:
            return
        records = []
        for ticker, df in data.items():
            for date, row in df.iterrows():
                records.append((
                    ticker,
                    date.strftime("%Y-%m-%d"),
                    round(float(row.get("Open", 0) or 0), 4),
                    round(float(row.get("High", 0) or 0), 4),
                    round(float(row.get("Low", 0) or 0), 4),
                    round(float(row["Close"]), 4),
                    int(row.get("Volume", 0) or 0),
                ))
        with self._connect() as conn:
            self._upsert(conn, records)
        logger.debug("Stored %d rows for %d tickers", len(records), len(data))

    def get_wide(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Return wide DataFrame matching _batch_download() format:
        rows=dates, columns=CLOSE_{ticker} and VOL_{ticker}."""
        placeholders = ",".join("?" * len(tickers))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT ticker, date, close, volume FROM ohlcv "
                f"WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ? "
                f"ORDER BY date",
                tickers + [start_date, end_date],
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="ticker", values=["close", "volume"])
        df.columns = [f"{'CLOSE' if field == 'close' else 'VOL'}_{ticker}"
                      for field, ticker in df.columns]
        return df

    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Return narrow OHLCV DataFrame for one ticker (for LLM analyst use)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM ohlcv "
                "WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
                (ticker, start_date, end_date),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df


# Module-level singleton shared across all callers in the same process
_cache = OHLCVCache()
