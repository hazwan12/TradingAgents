#!/usr/bin/env python3
"""Market universe pre-screener — filters all US-listed stocks to LLM-analysis candidates.

Two-stage pipeline
------------------
Stage 1 (fast):   Download full US ticker universe from NASDAQ, apply price/liquidity filters.
Stage 2 (signals): Batch-download 65 days of OHLCV, compute RSI / momentum / volume signals.
Output:            Ranked shortlist of top-N tickers, ready to feed into nightly_analysis.py.

Usage
-----
    python scripts/pre_screener.py                          # top 20 to console + JSON
    python scripts/pre_screener.py --top 50                 # widen to 50 candidates
    python scripts/pre_screener.py --min-price 10           # stricter price floor
    python scripts/pre_screener.py --min-volume 1000000     # stricter liquidity floor
    python scripts/pre_screener.py --dry-run                # show universe size, don't download prices
    python scripts/pre_screener.py --run-analysis           # run nightly_analysis on results afterwards

Scheduling (cron, runs 3:45 PM ET Mon-Fri, before nightly analysis at 3:55):
    45 15 * * 1-5  cd /path/to/TradingAgents && python scripts/pre_screener.py --run-analysis
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich import box

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

console = Console()

TRADINGAGENTS_HOME = Path.home() / ".tradingagents"
CACHE_DIR = TRADINGAGENTS_HOME / "cache" / "screener"
REPORTS_DIR = Path("reports")

# NASDAQ FTP — pipe-delimited lists of all US-listed securities
NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# GitHub mirror — fallback when NASDAQ FTP is blocked (e.g. outside US networks)
GITHUB_ALL_TICKERS_URL = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"

# Curated index sources (fallback when NASDAQ FTP is unreachable)
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Universe cache TTL: 24 h (constituents change rarely)
UNIVERSE_CACHE_TTL_HOURS = 24

# Price data: fetch 65 trading days so RSI(14) and SMA(50) are stable
PRICE_LOOKBACK_DAYS = 65
FAST_FILTER_DAYS = 5

# Batch size for yfinance.download — keeps individual HTTP requests manageable
DOWNLOAD_BATCH_SIZE = 200

# Signal weights for composite score (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "momentum_20d": 0.35,
    "volume_ratio": 0.25,
    "rsi_health": 0.20,
    "trend": 0.20,
}


# ---------------------------------------------------------------------------
# Universe fetching
# ---------------------------------------------------------------------------


def _fetch_url(url: str, headers: dict | None = None) -> str:
    resp = requests.get(url, timeout=30, headers=headers or {})
    resp.raise_for_status()
    return resp.text


def _parse_nasdaq_listed(raw: str) -> list[str]:
    """Parse nasdaqlisted.txt — drop ETFs, test issues, warrant/unit symbols."""
    lines = raw.strip().splitlines()
    if lines and lines[-1].startswith("File Creation Time"):
        lines = lines[:-1]
    df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    mask = df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
    return df.loc[mask, "Symbol"].tolist()


def _parse_other_listed(raw: str) -> list[str]:
    """Parse otherlisted.txt (NYSE, AMEX, etc.) — same filters."""
    lines = raw.strip().splitlines()
    if lines and lines[-1].startswith("File Creation Time"):
        lines = lines[:-1]
    df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    mask = df["ACT Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
    return df.loc[mask, "ACT Symbol"].tolist()


def _fetch_github_all_tickers() -> list[str]:
    """Fetch US ticker list from GitHub mirror (rreichel3/US-Stock-Symbols)."""
    raw = _fetch_url(GITHUB_ALL_TICKERS_URL)
    tickers = [
        t.strip().upper()
        for t in raw.splitlines()
        if t.strip() and re.match(r"^[A-Z]{1,5}$", t.strip())
    ]
    return sorted(set(tickers))


def _fetch_full_universe() -> list[str]:
    """Fetch all US-listed stocks (~9,000 tickers).

    Tries NASDAQ FTP first. If that times out or is blocked (common outside the
    US), silently falls back to a GitHub-hosted mirror that covers the same
    universe.
    """
    try:
        nasdaq = _parse_nasdaq_listed(_fetch_url(NASDAQ_LISTED_URL))
        other = _parse_other_listed(_fetch_url(OTHER_LISTED_URL))
        return sorted(set(nasdaq + other))
    except Exception as ftp_err:
        console.print(
            f"[yellow]NASDAQ FTP unreachable ({type(ftp_err).__name__}), "
            "switching to GitHub mirror...[/yellow]"
        )
        tickers = _fetch_github_all_tickers()
        console.print(f"[dim]GitHub mirror: {len(tickers)} tickers loaded.[/dim]")
        return tickers


def _fetch_sp500() -> list[str]:
    """Fetch S&P 500 constituents from GitHub datasets CSV (~500 tickers)."""
    raw = _fetch_url(SP500_CSV_URL)
    df = pd.read_csv(StringIO(raw))
    return sorted(df["Symbol"].dropna().str.upper().tolist())


def _fetch_nasdaq100() -> list[str]:
    """Fetch Nasdaq 100 constituents from Wikipedia (~100 tickers)."""
    # Wikipedia returns 403 for requests with no User-Agent (pd.read_html(url) hits
    # this directly via urllib); fetch the HTML ourselves with a browser UA instead.
    raw = _fetch_url(WIKI_NASDAQ100_URL, headers={"User-Agent": "Mozilla/5.0"})
    tables = pd.read_html(StringIO(raw))
    # The components table on Wikipedia has a 'Ticker' or 'Symbol' column
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        for candidate in ("ticker", "symbol"):
            if candidate in cols:
                col = t.columns[cols.index(candidate)]
                return sorted(t[col].dropna().astype(str).str.upper().tolist())
    raise ValueError("Could not find Nasdaq 100 ticker column in Wikipedia tables")


UNIVERSE_SOURCES = {
    "full": ("All US-listed stocks via NASDAQ FTP (~9,000)", _fetch_full_universe),
    "sp500": ("S&P 500 constituents (~500)", _fetch_sp500),
    "nasdaq100": ("Nasdaq 100 constituents (~100)", _fetch_nasdaq100),
}


def fetch_universe(source: str = "full", force_refresh: bool = False) -> list[str]:
    """Return tickers for the chosen universe, cached for 24 h.

    source: 'full' | 'sp500' | 'nasdaq100' | 'sp500+nasdaq100'
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = source.replace("+", "_")
    cache_path = CACHE_DIR / f"universe_{cache_key}.json"

    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < UNIVERSE_CACHE_TTL_HOURS:
            data = json.loads(cache_path.read_text())
            label, _ = UNIVERSE_SOURCES.get(source.split("+")[0], ("?", None))
            console.print(
                f"[dim]Universe '{source}': {len(data['tickers'])} tickers (cached {age_hours:.0f}h ago)[/dim]"
            )
            return data["tickers"]

    # Handle combined sources (e.g. 'sp500+nasdaq100')
    parts = source.split("+")
    all_tickers: set[str] = set()
    for part in parts:
        part = part.strip()
        if part not in UNIVERSE_SOURCES:
            raise ValueError(f"Unknown universe source '{part}'. Choose from: {list(UNIVERSE_SOURCES)}")
        label, fetch_fn = UNIVERSE_SOURCES[part]
        console.print(f"[dim]Fetching '{part}' universe: {label}...[/dim]")
        try:
            tickers = fetch_fn()
            all_tickers.update(tickers)
            console.print(f"[dim]  → {len(tickers)} tickers[/dim]")
        except Exception as e:
            if part == "full":
                console.print(
                    f"[yellow]Warning: Could not fetch full NASDAQ universe ({e}).[/yellow]\n"
                    "[yellow]Tip: Use --universe sp500 or --universe sp500+nasdaq100 "
                    "if NASDAQ FTP is blocked in your environment.[/yellow]"
                )
                raise
            raise

    # Sanitize: US universe must be plain uppercase letters only (1–5 chars).
    # Strips exchange/market-category suffixes like .NCM, .TO, .NGS that leak
    # in from NASDAQ FTP or GitHub mirror caches built before filtering was added.
    tickers = sorted({t for t in all_tickers if re.match(r'^[A-Z]{1,5}$', t)})
    cache_path.write_text(json.dumps({"fetched_at": datetime.now().isoformat(), "source": source, "tickers": tickers}), encoding="utf-8")
    console.print(f"[dim]Universe '{source}': {len(tickers)} tickers cached.[/dim]")
    return tickers


# ---------------------------------------------------------------------------
# Fast liquidity pre-filter (5-day download)
# ---------------------------------------------------------------------------


def _batch_download(tickers: list[str], period: str, batch_size: int = DOWNLOAD_BATCH_SIZE) -> pd.DataFrame:
    """Return Close + Volume DataFrame for all tickers, using the incremental
    SQLite cache. Only downloads bars not already cached — subsequent runs for
    the same universe fetch only the 1–2 new trading days since last run.
    Falls back to direct yfinance if the cache returns no data."""
    try:
        from tradingagents.dataflows.ohlcv_cache import _cache

        # Use yesterday as end_date — today's US market data isn't available until
        # after 4 PM ET (midnight SGT), and we always run after market close anyway.
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        days = FAST_FILTER_DAYS if period == f"{FAST_FILTER_DAYS}d" else PRICE_LOOKBACK_DAYS
        # Wider window to ensure enough trading days after weekends/holidays
        start_date = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y-%m-%d")

        console.print(f"[dim]  Downloading price data (incremental cache)...[/dim]")
        _cache.ensure_cached(tickers, start_date, end_date, batch_size=batch_size)

        df = _cache.get_wide(tickers, start_date, end_date)
        if not df.empty:
            return df
        logger.warning("OHLCV cache returned empty — falling back to direct yfinance download")
    except Exception as exc:
        logger.warning("OHLCV cache error (%s) — falling back to direct yfinance download", exc)

    # Fallback: original direct yfinance download
    return _batch_download_direct(tickers, period, batch_size)


def _batch_download_direct(tickers: list[str], period: str, batch_size: int = DOWNLOAD_BATCH_SIZE) -> pd.DataFrame:
    """Original direct yfinance download — used as fallback when the cache fails."""
    frames = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading price data...", total=total_batches)

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i: i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if raw.empty:
                    progress.advance(task)
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw["Close"]
                    volume = raw["Volume"]
                else:
                    close = raw[["Close"]].rename(columns={"Close": batch[0]})
                    volume = raw[["Volume"]].rename(columns={"Volume": batch[0]})
                frames.append((close, volume))
            except Exception as e:
                logger.warning(f"Batch {i // batch_size + 1} failed: {e}")
            finally:
                progress.advance(task)

    if not frames:
        return pd.DataFrame()

    all_close = pd.concat([f[0] for f in frames], axis=1)
    all_volume = pd.concat([f[1] for f in frames], axis=1)
    all_close.columns = [f"CLOSE_{c}" for c in all_close.columns]
    all_volume.columns = [f"VOL_{c}" for c in all_volume.columns]
    return pd.concat([all_close, all_volume], axis=1)


def fast_filter(
    tickers: list[str],
    min_price: float = 5.0,
    min_avg_volume: int = 500_000,
) -> list[str]:
    """Download 5 days of data, return tickers that pass price + volume thresholds."""
    console.print(
        f"[dim]Fast filter: {len(tickers)} tickers, min_price=${min_price}, "
        f"min_avg_volume={min_avg_volume:,}[/dim]"
    )
    raw = _batch_download(tickers, period=f"{FAST_FILTER_DAYS}d")
    if raw.empty:
        return tickers  # fallback: don't filter

    passing = []
    for ticker in tickers:
        try:
            close_col = f"CLOSE_{ticker}"
            vol_col = f"VOL_{ticker}"
            if close_col not in raw.columns or vol_col not in raw.columns:
                continue
            last_price = raw[close_col].dropna().iloc[-1] if not raw[close_col].dropna().empty else 0
            avg_vol = raw[vol_col].dropna().mean() if not raw[vol_col].dropna().empty else 0
            if last_price >= min_price and avg_vol >= min_avg_volume:
                passing.append(ticker)
        except Exception:
            continue

    console.print(f"[dim]Fast filter result: {len(passing)} tickers pass ({len(tickers) - len(passing)} dropped).[/dim]")
    return passing


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI for the last row of a price series. Returns NaN if insufficient data."""
    delta = series.diff().dropna()
    if len(delta) < period:
        return float("nan")
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")


def _rsi_health_score(rsi: float) -> float:
    """Convert RSI to a 0-1 buy-signal score.

    Ideal buy zone: 45-65 (uptrend, not yet overbought)
    Overbought (>75): penalised
    Oversold (<30): too weak, small score
    """
    if np.isnan(rsi):
        return 0.5  # neutral
    if rsi < 20:
        return 0.1
    if rsi < 30:
        return 0.3
    if rsi < 45:
        return 0.5
    if rsi <= 65:
        return 1.0  # sweet spot
    if rsi <= 75:
        return 0.6
    return 0.2  # overbought


def compute_signals(price_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute per-ticker signals from the full price DataFrame."""
    records = []
    for ticker in tickers:
        close_col = f"CLOSE_{ticker}"
        vol_col = f"VOL_{ticker}"
        if close_col not in price_df.columns:
            continue

        close = price_df[close_col].dropna()
        volume = price_df[vol_col].dropna() if vol_col in price_df.columns else pd.Series(dtype=float)

        if len(close) < 20:
            continue

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
        close_5d = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
        close_20d = float(close.iloc[-21]) if len(close) >= 21 else float(close.iloc[0])

        mom_1d = (last_close / prev_close - 1) if prev_close > 0 else 0.0
        mom_5d = (last_close / close_5d - 1) if close_5d > 0 else 0.0
        mom_20d = (last_close / close_20d - 1) if close_20d > 0 else 0.0

        avg_vol_20d = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else 0.0
        last_vol = float(volume.iloc[-1]) if not volume.empty else 0.0
        vol_ratio = (last_vol / avg_vol_20d) if avg_vol_20d > 0 else 1.0

        rsi14 = _rsi(close)
        rsi_score = _rsi_health_score(rsi14)

        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
        above_50ma = last_close > sma50 if not np.isnan(sma50) else False
        above_200ma = last_close > sma200 if not np.isnan(sma200) else False
        trend_score = (0.5 * float(above_50ma)) + (0.5 * float(above_200ma))

        records.append(
            {
                "ticker": ticker,
                "last_close": round(last_close, 2),
                "momentum_1d": round(mom_1d * 100, 3),
                "momentum_5d": round(mom_5d * 100, 3),
                "momentum_20d": round(mom_20d * 100, 3),
                "volume_ratio": round(vol_ratio, 3),
                "rsi_14": round(rsi14, 1) if not np.isnan(rsi14) else None,
                "rsi_health_score": round(rsi_score, 3),
                "above_50ma": above_50ma,
                "above_200ma": above_200ma,
                "trend_score": round(trend_score, 3),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------


def rank_candidates(signals_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Add composite score, rank, return top-N."""
    if signals_df.empty:
        return signals_df

    def _pct_rank(col: str) -> pd.Series:
        return signals_df[col].rank(pct=True, na_option="bottom")

    signals_df = signals_df.copy()
    signals_df["score_momentum"] = _pct_rank("momentum_20d")
    signals_df["score_volume"] = _pct_rank("volume_ratio")
    signals_df["score_rsi"] = signals_df["rsi_health_score"]  # already 0-1
    signals_df["score_trend"] = signals_df["trend_score"]    # already 0-1

    w = SIGNAL_WEIGHTS
    signals_df["composite"] = (
        w["momentum_20d"] * signals_df["score_momentum"]
        + w["volume_ratio"] * signals_df["score_volume"]
        + w["rsi_health"] * signals_df["score_rsi"]
        + w["trend"] * signals_df["score_trend"]
    ).round(4)

    return (
        signals_df.sort_values("composite", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_results_table(df: pd.DataFrame):
    t = Table(
        title=f"Top {len(df)} Candidates",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )
    t.add_column("#", width=4, justify="right")
    t.add_column("Ticker", width=7, style="bold white")
    t.add_column("Price", justify="right", width=8)
    t.add_column("Mom 20d", justify="right", width=8)
    t.add_column("Mom 5d", justify="right", width=7)
    t.add_column("Vol Ratio", justify="right", width=10)
    t.add_column("RSI", justify="right", width=6)
    t.add_column("50MA", width=6)
    t.add_column("200MA", width=7)
    t.add_column("Score", justify="right", width=7)

    for rank, row in df.iterrows():
        mom20 = row["momentum_20d"]
        mom20_str = f"[green]+{mom20:.1f}%[/green]" if mom20 >= 0 else f"[red]{mom20:.1f}%[/red]"
        mom5 = row["momentum_5d"]
        mom5_str = f"[green]+{mom5:.1f}%[/green]" if mom5 >= 0 else f"[red]{mom5:.1f}%[/red]"
        rsi_val = row.get("rsi_14")
        rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else "—"
        above50 = "[green]Y[/green]" if row["above_50ma"] else "[red]N[/red]"
        above200 = "[green]Y[/green]" if row["above_200ma"] else "[red]N[/red]"

        t.add_row(
            str(rank + 1),
            row["ticker"],
            f"${row['last_close']:,.2f}",
            mom20_str,
            mom5_str,
            f"{row['volume_ratio']:.2f}x",
            rsi_str,
            above50,
            above200,
            f"{row['composite']:.3f}",
        )

    console.print()
    console.print(t)


def _write_output(df: pd.DataFrame, date: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"SCREENER_{date}.json"
    out_path.write_text(
        json.dumps(
            {
                "date": date,
                "generated_at": datetime.now().isoformat(),
                "tickers": df["ticker"].tolist(),
                "candidates": df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def screen(
    top_n: int = 20,
    min_price: float = 5.0,
    min_avg_volume: int = 500_000,
    universe_source: str = "full",
    dry_run: bool = False,
    date: str | None = None,
) -> dict:
    """Run the full pre-screening pipeline. Returns result dict."""
    date = date or datetime.now().strftime("%Y-%m-%d")

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TradingAgents — Market Pre-Screener[/bold cyan]\n"
            f"[dim]Universe: {universe_source}  |  Top {top_n} candidates[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Stage 1a: Fetch universe
    console.print(Rule("[bold]Stage 1 — Universe[/bold]"))
    universe = fetch_universe(source=universe_source)
    console.print(f"Universe size: [bold]{len(universe):,}[/bold] tickers")

    if dry_run:
        console.print(f"\n[yellow]DRY RUN — would filter to liquid stocks and rank top {top_n}.[/yellow]")
        console.print(
            f"  Min price:      ${min_price}\n"
            f"  Min avg volume: {min_avg_volume:,}\n"
            f"  Signals:        momentum (20d/5d/1d), volume ratio, RSI-14, 50/200 MA\n"
            f"  Weights:        {SIGNAL_WEIGHTS}\n"
        )
        return {"success": True, "dry_run": True, "universe_size": len(universe)}

    # Stage 1b: Fast liquidity filter
    console.print()
    console.print(Rule("[bold]Stage 2 — Liquidity filter[/bold]"))
    liquid = fast_filter(universe, min_price=min_price, min_avg_volume=min_avg_volume)
    console.print(f"Liquid tickers: [bold]{len(liquid):,}[/bold]")

    # Stage 2: Full price download for signal computation
    console.print()
    console.print(Rule("[bold]Stage 3 — Signal computation[/bold]"))
    full_data = _batch_download(liquid, period=f"{PRICE_LOOKBACK_DAYS}d")
    if full_data.empty:
        console.print("[red]Failed to download price data.[/red]")
        return {"success": False, "error": "price download failed"}

    console.print("[dim]Computing signals...[/dim]")
    signals = compute_signals(full_data, liquid)
    console.print(f"Signals computed: {len(signals)} tickers")

    # Stage 3: Rank and filter
    console.print()
    console.print(Rule("[bold]Stage 4 — Ranking[/bold]"))
    top = rank_candidates(signals, top_n=top_n)
    _print_results_table(top)

    # Write output
    out_path = _write_output(top, date)
    console.print()
    console.print(Rule("[bold green]Done[/bold green]"))
    console.print(f"Output saved: [cyan]{out_path}[/cyan]")
    console.print()
    console.print("[dim]Feed into nightly analysis:[/dim]")
    console.print(f"  [bold]python scripts/nightly_analysis.py --ticker-file {out_path}[/bold]")
    console.print()

    return {
        "success": True,
        "date": date,
        "universe_source": universe_source,
        "universe_size": len(universe),
        "liquid_count": len(liquid),
        "candidates": top["ticker"].tolist(),
        "output_file": str(out_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pre-screen the full US equity universe for LLM analysis candidates"
    )
    parser.add_argument("--top", type=int, default=20, metavar="N", help="Number of candidates to output (default: 20)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum stock price in $ (default: 5.0)")
    parser.add_argument("--min-volume", type=int, default=500_000, metavar="VOL", help="Minimum 5-day avg daily volume (default: 500000)")
    parser.add_argument(
        "--universe",
        default="full",
        metavar="SOURCE",
        help=(
            "Ticker universe to scan. Options:\n"
            "  full           All US-listed stocks via NASDAQ FTP (~9,000) [default]\n"
            "  sp500          S&P 500 constituents (~500)\n"
            "  nasdaq100      Nasdaq 100 constituents (~100)\n"
            "  sp500+nasdaq100  Union of S&P 500 and Nasdaq 100\n"
            "Use 'sp500' or 'sp500+nasdaq100' if NASDAQ FTP is blocked in your environment."
        ),
    )
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Analysis date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading prices")
    parser.add_argument("--run-analysis", action="store_true", help="Run nightly_analysis.py on results immediately after screening")
    parser.add_argument("--print", dest="print_file", metavar="FILE", help="Print an existing screener JSON report as a table and exit")
    args = parser.parse_args()

    if args.print_file:
        import json as _json
        with open(args.print_file) as _f:
            _data = _json.load(_f)
        _df = pd.DataFrame(_data["candidates"])
        _df.index = range(len(_df))
        _print_results_table(_df)
        sys.exit(0)

    result = screen(
        top_n=args.top,
        min_price=args.min_price,
        min_avg_volume=args.min_volume,
        universe_source=args.universe,
        dry_run=args.dry_run,
        date=args.date,
    )

    if not result.get("success"):
        sys.exit(1)

    if args.run_analysis and not args.dry_run:
        import subprocess
        output_file = result.get("output_file")
        console.print(Rule("[bold]Running nightly analysis on screener results[/bold]"))
        subprocess.run(
            [sys.executable, "scripts/nightly_analysis.py", "--ticker-file", output_file, "--date", args.date],
            check=False,
        )


if __name__ == "__main__":
    main()
