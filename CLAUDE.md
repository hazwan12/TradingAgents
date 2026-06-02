# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install .

# Run CLI — use python -m cli.main; the tradingagents entrypoint may not be
# on PATH in dev environments (editable install or cloned without pip install)
python -m cli.main

# Tests
pytest                         # full suite
pytest -m unit                 # unit tests only
pytest -m integration          # integration tests only
pytest -m smoke                # smoke tests only
pytest tests/test_env_overrides.py   # single file

# Docker
cp .env.example .env
docker compose run --rm tradingagents
docker compose --profile ollama run --rm tradingagents-ollama   # local models
```

## Setup from a fresh clone

```bash
git clone <repo-url>
cd TradingAgents
pip install .
cp .env.example .env
# Add your API key to .env (see LLM Provider Setup below)
python -m cli.main
```

## LLM Provider Setup

### Google Gemini (cloud, free tier)

Get a key at aistudio.google.com/apikey, then add to `.env`:
```
GOOGLE_API_KEY=your-key-here
```
**Important:** Use `Gemini 2.5 Flash` for both quick and deep models. `Gemini 3.1 Pro (preview)` and `Gemini 3.5 Flash` require prepayment credits and will fail with `429 RESOURCE_EXHAUSTED` on a fresh free-tier account.

### Ollama (local GPU — recommended for home PC)

Best option for local inference — no API key, no usage cost.

```bash
# Install Ollama (ollama.com/install)
ollama pull qwen2.5:14b        # or qwen3:8b for 8 GB VRAM
ollama serve                   # starts on localhost:11434
```

Choose model size by VRAM:
| VRAM | Model |
|---|---|
| 6–8 GB | `qwen2.5:7b` / `qwen3:8b` |
| 12 GB | `qwen2.5:14b` / `qwen3:14b` |
| 16–24 GB | `qwen2.5:32b` |
| 24 GB+ | `qwen2.5:72b` (quantized) |

In `.env`:
```
TRADINGAGENTS_LLM_PROVIDER=ollama
TRADINGAGENTS_DEEP_THINK_LLM=qwen2.5:14b
TRADINGAGENTS_QUICK_THINK_LLM=qwen2.5:7b
```
Then just run `python -m cli.main` — no API key needed.

## Architecture

TradingAgents is a multi-agent LLM trading analysis framework built on LangGraph. It decomposes financial analysis into specialized agents that collaborate through a structured workflow.

### Execution flow

```
Analyst Phase → Research Debate → Trader → Risk Debate → Portfolio Manager
```

1. **Analyst Phase** — parallel or sequential depending on config. Each analyst uses tool-calling LLM nodes to fetch and interpret data (market, sentiment, news, fundamentals).
2. **Research Debate** — Bull and Bear researchers argue over the analysts' reports. Research Manager synthesizes to a `ResearchPlan`.
3. **Trader** — converts the `ResearchPlan` into a concrete `TraderProposal`.
4. **Risk Debate** — Aggressive, Conservative, Neutral analysts debate risk. Portfolio Manager issues a final `PortfolioDecision` (Buy/Overweight/Hold/Underweight/Sell).
5. **Memory** — decision written to `~/.tradingagents/memory/trading_memory.md`; next run adds a reflection based on actual return.

### Key directories

- `tradingagents/agents/` — all agent implementations (analysts, researchers, managers, trader, risk_mgmt)
- `tradingagents/graph/` — LangGraph orchestration: `trading_graph.py` (main entry), `setup.py` (graph wiring), `conditional_logic.py` (routing)
- `tradingagents/dataflows/` — data vendor abstraction; `interface.py` routes to yfinance or Alpha Vantage
- `tradingagents/llm_clients/` — multi-provider LLM abstraction; `factory.py` creates clients, `model_catalog.py` lists available models
- `tradingagents/default_config.py` — all tuneable settings with env-var overrides
- `cli/` — interactive Typer/Rich TUI

### Agent state

Agents communicate through a shared `AgentState` (LangGraph `MessagesState` subclass defined in `agents/agent_states.py`). Debate phases maintain separate history fields (`bull_history`, `bear_history`, `invest_debate_state`, `risk_debate_state`).

### Structured outputs

`agents/schemas.py` defines Pydantic models: `ResearchPlan`, `TraderProposal`, `PortfolioDecision`. Binding strategy differs per provider (`agents/utils/structured.py`): `json_schema` for OpenAI, `response_schema` for Gemini, tool-use for Anthropic. Free-text fallback when the provider doesn't support structured output.

### LLM configuration

Every run uses two LLMs: `deep_think_llm` (reasoning-heavy tasks: managers, trader) and `quick_think_llm` (data-fetching agents). The factory in `llm_clients/factory.py` lazily loads provider modules. Supported providers: openai, google, anthropic, xai, deepseek, qwen, glm, minimax, openrouter, ollama, azure.

### Configuration and environment variables

`DEFAULT_CONFIG` in `tradingagents/default_config.py` is the canonical settings dict. All keys can be overridden via `TRADINGAGENTS_<KEY>` env vars (e.g., `TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_MAX_DEBATE_ROUNDS`). Provider API keys use standard names: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, etc. See `.env.example` for a full list.

### Persistence

- **Memory log**: `~/.tradingagents/memory/trading_memory.md` — append-only markdown. Each entry starts as `status: pending`; the next run with the same ticker adds a reflection with actual return and alpha vs. benchmark.
- **Checkpoints**: `~/.tradingagents/cache/checkpoints/<TICKER>.db` (SQLite, opt-in via `checkpoint_enabled`). Cleared on successful completion.
- **Results**: `~/.tradingagents/logs/` — JSON state snapshots per run.

### Programmatic usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-opus-4-8"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

### Multi-market support

Non-US tickers use suffix conventions: `.NS` (India NSE), `.T` (Tokyo), `.HK` (Hong Kong), `.SS`/`.SZ` (China A-shares), `.L` (London). Crypto uses `-USD` (e.g., `BTC-USD`). Forex uses standard pairs (e.g., `EURUSD=X`). The instrument identity resolver in `agents/utils/agent_utils.py` → `resolve_instrument_identity` normalizes tickers and prevents wrong-company hallucination.

### Testing

Tests are under `tests/` with markers `unit`, `integration`, `smoke`. Notable test files: `test_env_overrides.py`, `test_checkpoint_resume.py`, `test_memory_log.py`, `test_structured_agents.py`, `test_instrument_identity.py`. Most tests mock LLM calls and data fetches so no API keys are required.

## Planned features (not yet built)

### 1. Watchlist scanner (`scripts/scan_watchlist.py`)

The framework analyzes one ticker at a time. A wrapper that loops over a watchlist and produces a ranked summary table is planned but not yet implemented. Skeleton:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(config={**DEFAULT_CONFIG, "llm_provider": "ollama", ...})
watchlist = ["NVDA", "AAPL", "MSFT"]
results = {ticker: ta.propagate(ticker, date) for ticker in watchlist}
```

Each ticker takes ~10 minutes and ~500k tokens. On free-tier Gemini, scanning more than 2–3 tickers will likely hit daily quota. Use Ollama locally for unrestricted batch runs.

### 2. Portfolio advisor (`scripts/portfolio_advisor.py`)

Not built yet. The `TraderProposal` schema has `position_sizing` (e.g. "5% of portfolio") and `entry_price`, but the framework has no awareness of actual holdings or cash.

Planned inputs: current cash budget + holdings dict `{"NVDA": 10, "AAPL": 5}`.
Planned output: specific buy/sell unit counts derived from rating + position sizing % × total portfolio value ÷ current price.

`PortfolioDecision` and `TraderProposal` are the output schemas to read from — both in `tradingagents/agents/schemas.py`.
