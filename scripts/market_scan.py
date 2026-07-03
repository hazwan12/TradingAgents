#!/usr/bin/env python3
"""Unified, cached, incremental market scan shared by the personal pipeline and the
channel broadcast.

Screens each requested universe (full/sp500/nasdaq100/...) for its top-N technical
candidates, unions them by ticker (tagging which universe(s) each came from, plus any
caller-supplied extra tickers like held positions or open channel positions), then runs
one LLM WatchlistScanner pass over the unioned ticker list.

Results are cached per date at reports/UNIFIED_SCAN_{date}.json with two levels of
checkpointing so a crashed run can resume without starting from scratch:

  Checkpoint 1 — Screener (2a-2c): cache is written after each universe's technical
  screening completes, so a restart skips re-downloading OHLCV for already-screened
  universes.

  Checkpoint 2 — LLM analysis (2d): tickers are processed in batches and the cache is
  written after each batch, so a restart picks up from the last completed batch rather
  than re-analyzing all tickers from scratch.

A second call on the same day only analyzes tickers not already present in the cache,
so whichever consumer (personal pipeline, channel broadcast) runs second pays no extra
LLM cost for tickers the first one already covered.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pre_screener import screen
from scripts.scan_watchlist import WatchlistScanner
from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("reports")

# Number of tickers to LLM-analyze per batch before writing a checkpoint.
# Lower = more frequent saves, finer resume granularity, slightly more disk I/O.
# With max_workers=3, a batch of 5 is ~2 parallel rounds before each checkpoint.
CHECKPOINT_BATCH_SIZE = 5


def _cache_path(date: str) -> Path:
    return RESULTS_DIR / f"UNIFIED_SCAN_{date}.json"


def _load_cache(date: str) -> Optional[dict]:
    path = _cache_path(date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(date: str, cache: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    _cache_path(date).write_text(json.dumps(cache, indent=2), encoding="utf-8")


def run_unified_scan(
    date: str,
    universes: List[str] = ("full", "sp500", "nasdaq100"),
    top_n: int = 20,
    extra_tickers: Optional[Dict[str, List[str]]] = None,
    config: Optional[Dict[str, Any]] = None,
    secondary_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 5,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Run (or reuse cached) unified scan across universes + extra tickers.

    Returns:
        {
          "date": str,
          "universes": list[str],
          "top_n": int,
          "candidates_by_universe": {universe: [tickers]},
          "tags": {ticker: [origin tags, e.g. "sp500", "held", "channel_open"]},
          "results": {ticker: {rating, action, executive_summary, ...}},
          "failed": [(ticker, error), ...],
        }
    """
    universes = list(universes)
    extra_tickers = extra_tickers or {}
    config = config or DEFAULT_CONFIG.copy()

    secondary_model = (secondary_config or {}).get("deep_think_llm", "")

    # Load existing cache if available and config matches.
    cache = None if force_refresh else _load_cache(date)
    if cache and cache.get("universes") == universes and cache.get("top_n") == top_n:
        logger.info(f"Resuming from cached unified scan for {date} "
                    f"({len(cache.get('results', {}))} tickers already analyzed)")
    else:
        cache = {
            "date": date,
            "universes": universes,
            "top_n": top_n,
            "candidates_by_universe": {},
            "tags": {},
            "results": {},
            "secondary_results": {},
            "failed": [],
        }

    # --- Checkpoint 1: Screener (2a-2c) ---
    # Run screen() only for universes not yet in the cache. Write the cache after
    # each universe so a restart skips already-screened universes.
    for universe in universes:
        if universe in cache["candidates_by_universe"]:
            candidates = cache["candidates_by_universe"][universe]
            logger.info(f"Checkpoint: skipping screener for universe={universe} "
                        f"({len(candidates)} candidates already cached)")
            # Re-apply tags in case they were lost (e.g. force_refresh partial)
            for ticker in candidates:
                cache["tags"].setdefault(ticker, [])
                if universe not in cache["tags"][ticker]:
                    cache["tags"][ticker].append(universe)
            continue

        screen_result = screen(top_n=top_n, universe_source=universe, date=date)
        if not screen_result.get("success"):
            logger.warning(f"Screener failed for universe={universe}, skipping")
            cache["candidates_by_universe"][universe] = []
        else:
            candidates = screen_result["candidates"]
            cache["candidates_by_universe"][universe] = candidates
            for ticker in candidates:
                cache["tags"].setdefault(ticker, [])
                if universe not in cache["tags"][ticker]:
                    cache["tags"][ticker].append(universe)

        # Checkpoint: persist screener results immediately so a restart skips this universe
        _save_cache(date, cache)

    # Merge in extra tickers (held positions, open channel positions, ...)
    for tag, tickers in extra_tickers.items():
        for ticker in tickers:
            cache["tags"].setdefault(ticker, [])
            if tag not in cache["tags"][ticker]:
                cache["tags"][ticker].append(tag)

    # --- Checkpoint 2: LLM analysis (2d) ---
    # Process missing tickers in batches, writing the cache after each batch so a
    # restart resumes from the last completed batch rather than from ticker 0.
    all_tickers = set(cache["tags"].keys())
    already_analyzed = set(cache["results"].keys())
    missing = sorted(all_tickers - already_analyzed)

    if not missing:
        logger.info("No new tickers to analyze — unified scan fully satisfied by cache")
    else:
        logger.info(f"[Primary: {config.get('deep_think_llm', 'default')}] "
                    f"Analyzing {len(missing)} ticker(s) in batches of {CHECKPOINT_BATCH_SIZE} "
                    f"({len(already_analyzed)} already cached)")
        scanner = WatchlistScanner(max_workers=max_workers, config=config)

        for batch_start in range(0, len(missing), CHECKPOINT_BATCH_SIZE):
            batch = missing[batch_start: batch_start + CHECKPOINT_BATCH_SIZE]
            batch_num = batch_start // CHECKPOINT_BATCH_SIZE + 1
            total_batches = (len(missing) + CHECKPOINT_BATCH_SIZE - 1) // CHECKPOINT_BATCH_SIZE
            logger.info(f"Primary batch {batch_num}/{total_batches}: {', '.join(batch)}")

            batch_result = scanner.scan(batch, date)
            for item in batch_result["results"]:
                item["primary_model"] = config.get("deep_think_llm", "")
                cache["results"][item["ticker"]] = item
            cache["failed"].extend(batch_result["failed"])

            _save_cache(date, cache)
            logger.info(f"Primary checkpoint — {len(cache['results'])}/{len(all_tickers)} tickers complete")

    # --- Secondary model scan (optional) ---
    if secondary_config:
        cache.setdefault("secondary_results", {})
        cache["primary_model"] = config.get("deep_think_llm", "")
        cache["secondary_model"] = secondary_model

        secondary_missing = sorted(all_tickers - set(cache["secondary_results"].keys()))
        if not secondary_missing:
            logger.info("Secondary scan fully satisfied by cache")
        else:
            logger.info(f"[Secondary: {secondary_model}] "
                        f"Analyzing {len(secondary_missing)} ticker(s) "
                        f"({len(cache['secondary_results'])} already cached)")
            sec_scanner = WatchlistScanner(max_workers=max_workers, config=secondary_config)

            for batch_start in range(0, len(secondary_missing), CHECKPOINT_BATCH_SIZE):
                batch = secondary_missing[batch_start: batch_start + CHECKPOINT_BATCH_SIZE]
                batch_num = batch_start // CHECKPOINT_BATCH_SIZE + 1
                total_batches = (len(secondary_missing) + CHECKPOINT_BATCH_SIZE - 1) // CHECKPOINT_BATCH_SIZE
                logger.info(f"Secondary batch {batch_num}/{total_batches}: {', '.join(batch)}")

                batch_result = sec_scanner.scan(batch, date)
                for item in batch_result["results"]:
                    item["secondary_model"] = secondary_model
                    cache["secondary_results"][item["ticker"]] = item

                _save_cache(date, cache)
                logger.info(f"Secondary checkpoint — {len(cache['secondary_results'])}/{len(all_tickers)} tickers complete")

    return cache
