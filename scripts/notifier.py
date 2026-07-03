"""Telegram notifications for nightly analysis results.

Two independent destinations:
    - Personal DM (TELEGRAM_CHAT_ID) — your own holdings-based recommendations.
    - Group broadcast (TELEGRAM_BROADCAST_CHAT_ID) — generic top-10-per-universe
      picks, same content for every member of the group.

Setup:
    1. Create a bot via @BotFather on Telegram, get the bot token.
    2. Message your bot once, then visit
       https://api.telegram.org/bot<TOKEN>/getUpdates to find your personal chat id.
    3. For the broadcast: create/use a Telegram group, add the bot to it, send a
       message in the group, then check getUpdates again for a chat id (groups have
       negative ids).
    4. Add to .env:
         TELEGRAM_BOT_TOKEN=...
         TELEGRAM_CHAT_ID=...
         TELEGRAM_BROADCAST_CHAT_ID=...
"""

import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def telegram_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def broadcast_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_BROADCAST_CHAT_ID"))


def _post(token: str, chat_id: str, message: str) -> bool:
    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=token),
            data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.warning(f"Failed to send Telegram notification to {chat_id}: {e}")
        return False


def send_telegram(message: str) -> bool:
    """Send a message to the personal chat. Returns True on success, False if unconfigured or failed."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID unset) — skipping notification")
        return False
    return _post(token, chat_id, message)


def send_telegram_photo(image_bytes: bytes, caption: str = "", broadcast: bool = False) -> bool:
    """Send a photo to the personal chat or broadcast group."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_BROADCAST_CHAT_ID" if broadcast else "TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.warning(f"Failed to send Telegram photo to {chat_id}: {e}")
        return False


def send_telegram_broadcast(message: str) -> bool:
    """Send a message to the broadcast group. Returns True on success, False if unconfigured or failed."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_BROADCAST_CHAT_ID")
    if not token or not chat_id:
        logger.info("Telegram broadcast not configured (TELEGRAM_BROADCAST_CHAT_ID unset) — skipping broadcast")
        return False
    return _post(token, chat_id, message)


def _suggest_order_type(
    action: str,
    current_price: float | None,
    entry_price: float | None,
    stop_loss: float | None,
    price_target: float | None,
) -> list[str]:
    """Return Moomoo order type recommendation lines based on price levels.

    BUY logic:
      - No entry price → Market order (no specific level to target)
      - Entry within ±2% of current → Limit at entry (fill at or better)
      - Entry meaningfully below current → Limit-If-Touched (auto-triggers on dip)
      - Entry meaningfully above current → Stop Limit (buy on upside breakout)

    SELL logic:
      - Market order to exit, with stop-loss context if available.
    """
    lines = []
    if action == "Buy":
        if not entry_price or not current_price:
            lines.append("Order type: *Market* (no specific entry level — buy at current price)")
        else:
            diff_pct = (entry_price - current_price) / current_price * 100
            if abs(diff_pct) <= 2:
                lines.append(f"Order type: *Limit* @ ${entry_price:,.2f} (at or better than entry)")
            elif diff_pct < -2:
                lines.append(
                    f"Order type: *Limit-If-Touched* @ ${entry_price:,.2f} "
                    f"(auto-triggers if price dips {abs(diff_pct):.1f}% to entry)"
                )
            else:
                lines.append(
                    f"Order type: *Stop Limit* @ ${entry_price:,.2f} "
                    f"(auto-triggers on breakout {diff_pct:.1f}% above current)"
                )
        if stop_loss:
            lines.append(f"Risk mgmt: Set *Stop Loss* @ ${stop_loss:,.2f} once filled")
        if price_target:
            lines.append(f"Take profit: *Limit-If-Touched* @ ${price_target:,.2f}")

    elif action == "Sell":
        lines.append("Order type: *Market* to exit, or *Limit* slightly below current price")
        if stop_loss:
            lines.append(f"Already breached stop @ ${stop_loss:,.2f} — exit promptly")

    return lines


def _return_emoji(pct: float) -> str:
    if pct >= 10:
        return "🔥"
    if pct > 0:
        return "📈"
    return "📉"


def _model_rating_line(primary_rating: str, primary_model: str,
                        secondary_rating: str, secondary_model: str) -> str:
    """Build a compact dual-model rating line, e.g. 'qwen: Buy  |  deepseek: Hold'."""
    p_label = primary_model.split(":")[0] if primary_model else "Primary"
    s_label = secondary_model.split(":")[0] if secondary_model else "Secondary"
    if secondary_rating:
        return f"{p_label}: {primary_rating}  |  {s_label}: {secondary_rating}"
    return f"{p_label}: {primary_rating}"


_SEPARATOR = "────────────"


def _is_meaningful(text: str) -> bool:
    """Return True only when text contains real content, not just punctuation."""
    return bool(text) and len(text.strip()) > 3 and not all(c in '.?!,:;-_ ' for c in text.strip())


_RECOMMENDATION_RE = re.compile(r"recommendation\**:\**\s*(\w+)", re.IGNORECASE)
_BULLISH = {"buy", "overweight"}
_BEARISH = {"sell", "underweight"}


def _contradicts_action(action: str, verdict: str) -> bool:
    """True if verdict's embedded recommendation conflicts with the action taken.

    judge_verdict comes from the Research Manager's debate judgment, an earlier
    pipeline stage that the Trader/Portfolio Manager can override — so it can
    disagree with the final action. Suppress it rather than show a contradiction.
    """
    match = _RECOMMENDATION_RE.search(verdict or "")
    if not match:
        return False
    rec = match.group(1).lower()
    action_l = (action or "").strip().lower()
    if action_l in _BULLISH and (rec in _BEARISH or rec == "hold"):
        return True
    if action_l in _BEARISH and (rec in _BULLISH or rec == "hold"):
        return True
    return False


def _reasoning_lines(bull: str, bear: str, verdict: str, action: str = "") -> list[str]:
    lines = []
    if _is_meaningful(bull):
        lines.append(f"🐂 *Bull:* {bull}")
    if _is_meaningful(bear):
        lines.append(f"🐻 *Bear:* {bear}")
    if verdict and not _contradicts_action(action, verdict):
        lines.append(f"🔎 *Research debate call:* {verdict}")
    return lines


def format_orders_message(date: str, orders_doc: dict, result: dict) -> str:
    orders = orders_doc.get("orders", [])
    lines = [f"*TradingAgents — Personal — {date}*", ""]
    lines.append(f"Portfolio value: ${result['portfolio_value']:,.2f}")
    lines.append(f"Cash available:  ${result['cash_available']:,.2f}")
    lines.append("")

    if not orders:
        lines.append("No actionable orders — all positions Hold.")
        return "\n".join(lines)

    sorted_orders = sorted(orders, key=lambda o: o["ticker"])
    lines.append(f"*{len(sorted_orders)} staged order(s):*")
    for idx, o in enumerate(sorted_orders, 1):
        if idx > 1:
            lines.append(_SEPARATOR)
        arrow = "▲ BUY" if o["direction"] == "buy" else "▼ SELL"
        lines.append("")
        lines.append(f"[{idx}/{len(sorted_orders)}] {arrow} *{o['ticker']}*  {o['units']:.2f} units")
        lines.append(
            "  " + _model_rating_line(
                o.get("rating", ""), o.get("primary_model", ""),
                o.get("secondary_rating", ""), o.get("secondary_model", ""),
            )
        )
        lines.append(
            f"  Est. price: ${o['estimated_price']:,.2f}  "
            f"Limit: ${o['limit_price']:,.4f}  "
            f"Total: ${o['estimated_total']:,.2f}"
        )
        for ol in _suggest_order_type(
            action=o["direction"].capitalize(),
            current_price=o.get("estimated_price"),
            entry_price=None, stop_loss=None, price_target=None,
        ):
            lines.append(f"  {ol}")
        for rl in _reasoning_lines(
            o.get("bull_thesis", ""), o.get("bear_concern", ""), o.get("judge_verdict", ""),
            action=o["direction"].capitalize(),
        ):
            lines.append(f"  {rl}")
        if o.get("executive_summary"):
            lines.append(f"  _{o['executive_summary']}_")

    # Holding power summary — total downside exposure across all buy orders
    buy_orders = [o for o in orders if o["direction"] == "buy" and o.get("max_drawdown_exposure")]
    if buy_orders:
        total_exposure = sum(o["max_drawdown_exposure"] for o in buy_orders)
        lines.append("")
        lines.append("*💰 Holding Power Check:*")
        for o in sorted(buy_orders, key=lambda o: o["ticker"]):
            lines.append(
                f"  {o['ticker']}: max drawdown ${o['max_drawdown_exposure']:,.2f} "
                f"({o['drawdown_pct']}% from entry to stop)"
            )
        lines.append(f"  *Total downside exposure: ${total_exposure:,.2f}*")
        cash = result.get("cash_available", 0)
        if cash > 0:
            exposure_pct = round(total_exposure / result["portfolio_value"] * 100, 1)
            lines.append(f"  _{exposure_pct}% of portfolio value — ensure you can absorb this before executing._")

    lines.append("")
    lines.append("Review and submit before market open.")
    return "\n".join(lines)


def format_failure_message(date: str, error: str) -> str:
    return f"*TradingAgents — {date}*\n\nAnalysis failed: {error}"


def format_holdings_digest(date: str, open_positions: list[dict], stats: dict | None = None) -> str:
    """Daily digest of all currently open positions with P&L and track record.
    Sent even when nothing changed so subscribers who joined late have full context."""
    lines = [f"*📊 Open Positions — {date}*", ""]

    if not open_positions:
        lines.append("No open positions.")
        return "\n".join(lines)

    total = len(open_positions)
    for idx, r in enumerate(sorted(open_positions, key=lambda r: r["ticker"]), 1):
        if idx > 1:
            lines.append(_SEPARATOR)
        t = r["ticker"]
        days = r.get("days_held")
        opened = r.get("opened_date", "")
        current_price = r.get("current_price")
        entry_price = r.get("entry_price")
        stop_loss = r.get("stop_loss")
        price_target = r.get("price_target")
        rating = r.get("rating", "Hold")

        # Header: numbered ticker + how long held
        day_label = ("Day 1" if days == 0 else f"Day {days + 1}") if days is not None else ""
        header = f"[{idx}/{total}] *{t}* — {rating}"
        if day_label and opened:
            header += f"  _({day_label}, opened {opened})_"
        lines.append(header)

        # Model ratings if available
        model_line = _model_rating_line(
            rating, r.get("primary_model", ""),
            r.get("secondary_rating", ""), r.get("secondary_model", ""),
        )
        if r.get("primary_model"):
            lines.append(model_line)

        # P&L line
        return_pct = r.get("return_pct")
        if entry_price and current_price:
            emoji = _return_emoji(return_pct) if return_pct is not None else ""
            sign = "+" if return_pct and return_pct >= 0 else ""
            ret_str = f"{sign}{return_pct:.1f}% {emoji}" if return_pct is not None else ""
            lines.append(
                f"Entry: ${entry_price:,.2f} | Now: ${current_price:,.2f}"
                + (f" | {ret_str}" if ret_str else "")
            )
        elif current_price:
            lines.append(f"Current: ${current_price:,.2f}")

        # Stop / target
        price_parts = []
        if stop_loss:
            price_parts.append(f"Stop: ${stop_loss:,.2f}")
        if price_target:
            price_parts.append(f"Target: ${price_target:,.2f}")
        if price_parts:
            lines.append("  ".join(price_parts))

        # Order type recommendation
        for ol in _suggest_order_type(
            "Buy", current_price, entry_price, stop_loss, price_target
        ):
            lines.append(ol)

    lines.append("")
    # Track record block
    if stats and stats.get("total_trades", 0) > 0:
        n = stats["total_trades"]
        avg = stats.get("cumulative_return_pct", 0)
        wr = stats.get("win_rate_pct", 0)
        sign = "+" if avg >= 0 else ""
        lines.append(f"📈 *Track Record*  ({n} closed trade{'s' if n != 1 else ''})")
        lines.append(f"Avg return: {sign}{avg:.1f}% | Win rate: {wr:.0f}%")
        lines.append("")

    lines.append("_Positions close on a Sell signal. Not financial advice._")
    return "\n".join(lines).rstrip()


def format_position_update_message(date: str, opened: list[dict], closed: list[dict]) -> str:
    """Virtual-portfolio BUY/SELL events for the channel. Returns '' if nothing to announce."""
    if not opened and not closed:
        return ""

    lines = [f"*TradeAgents Signals — {date}*", ""]

    if opened:
        lines.append(f"*▲ New BUY ({len(opened)}):*")
        lines.append("")
        for idx, r in enumerate(sorted(opened, key=lambda r: r["ticker"]), 1):
            if idx > 1:
                lines.append(_SEPARATOR)
            universes = ", ".join(r.get("universes", []))
            lines.append(f"[{idx}/{len(opened)}] *{r['ticker']}*  _({universes})_")
            lines.append(
                _model_rating_line(
                    r.get("rating", ""), r.get("primary_model", ""),
                    r.get("secondary_rating", ""), r.get("secondary_model", ""),
                )
            )

            current_price = r.get("current_price")
            entry_price = r.get("entry_price")
            stop_loss = r.get("stop_loss")
            price_target = r.get("price_target")

            price_parts = []
            if current_price:
                price_parts.append(f"Current: ${current_price:,.2f}")
            if entry_price:
                price_parts.append(f"Entry: ${entry_price:,.2f}")
            if stop_loss:
                price_parts.append(f"Stop: ${stop_loss:,.2f}")
            if price_target:
                price_parts.append(f"Target: ${price_target:,.2f}")
            if price_parts:
                lines.append("  ".join(price_parts))

            for ol in _suggest_order_type("Buy", current_price, entry_price, stop_loss, price_target):
                lines.append(ol)

            for rl in _reasoning_lines(
                r.get("bull_thesis", ""), r.get("bear_concern", ""), r.get("judge_verdict", ""),
                action="Buy",
            ):
                lines.append(rl)

            if r.get("executive_summary"):
                lines.append(f"_{r['executive_summary']}_")
        lines.append("")

    if closed:
        lines.append(f"*▼ SELL — close position ({len(closed)}):*")
        lines.append("")
        for idx, r in enumerate(sorted(closed, key=lambda r: r["ticker"]), 1):
            if idx > 1:
                lines.append(_SEPARATOR)
            days = r.get("days_held", 0)
            day_str = f"held {days} day{'s' if days != 1 else ''}" if days else ""
            lines.append(f"[{idx}/{len(closed)}] *{r['ticker']}*" + (f"  _({day_str})_" if day_str else ""))
            lines.append(
                _model_rating_line(
                    r.get("rating", ""), r.get("primary_model", ""),
                    r.get("secondary_rating", ""), r.get("secondary_model", ""),
                )
            )
            # Exit P&L
            entry_price = r.get("entry_price")
            exit_price = r.get("current_price") or r.get("exit_price")
            return_pct = r.get("return_pct")
            if entry_price and exit_price:
                emoji = _return_emoji(return_pct) if return_pct is not None else ""
                sign = "+" if return_pct and return_pct >= 0 else ""
                ret_str = f"{sign}{return_pct:.1f}% {emoji}" if return_pct is not None else ""
                lines.append(
                    f"Exit: ${exit_price:,.2f} | Entry was: ${entry_price:,.2f}"
                    + (f" | *{ret_str}*" if ret_str else "")
                )
            current_price = r.get("current_price")
            for ol in _suggest_order_type("Sell", current_price, None, r.get("stop_loss"), None):
                lines.append(ol)
            for rl in _reasoning_lines(
                r.get("bull_thesis", ""), r.get("bear_concern", ""), r.get("judge_verdict", ""),
                action="Sell",
            ):
                lines.append(rl)
            if r.get("executive_summary"):
                lines.append(f"_{r['executive_summary']}_")
            lines.append("")

    lines.append("_Not financial advice. Do your own due diligence._")
    return "\n".join(lines).rstrip()
