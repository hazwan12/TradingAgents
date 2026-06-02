"""Portfolio calculation utilities for position sizing and rebalancing."""

import re
from typing import Dict, Optional, Tuple

DEFAULT_POSITION_SIZE_PCT = 0.03  # 3% fallback allocation


def parse_position_sizing(text: str) -> float:
    """Extract position sizing percentage from text.

    Handles formats like:
    - "5% of portfolio"
    - "allocate 5%"
    - "3-5% of portfolio" (uses midpoint)
    - Fallback to DEFAULT_POSITION_SIZE_PCT if unparseable

    Args:
        text: Text containing position sizing guidance

    Returns:
        Float percentage as decimal (e.g., 0.05 for 5%)
    """
    if not text:
        return DEFAULT_POSITION_SIZE_PCT

    # Clean markdown formatting
    text = re.sub(r"[\*_]{1,2}", "", text)

    # Look for single percentage: "5%"
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        pct = float(match.group(1)) / 100.0
        return pct

    # Look for range: "3-5%" → use midpoint
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%", text)
    if match:
        min_pct = float(match.group(1))
        max_pct = float(match.group(2))
        midpoint = (min_pct + max_pct) / 2.0 / 100.0
        return midpoint

    # Fallback: use default
    return DEFAULT_POSITION_SIZE_PCT


def calculate_units_to_trade(
    current_price: float,
    target_allocation_pct: float,
    portfolio_value: float,
    current_units: float = 0.0,
) -> Tuple[float, float]:
    """Calculate units to buy/sell and dollar amount.

    Formula:
    - Target allocation in $ = target_allocation_pct × total_portfolio_value
    - Current holding value = current_units × current_price
    - Rebalance amount = target_allocation - current_holding_value
    - Units to trade = rebalance_amount / current_price

    Args:
        current_price: Current market price per unit
        target_allocation_pct: Target allocation as decimal (e.g., 0.05 for 5%)
        portfolio_value: Total portfolio value (cash + holdings)
        current_units: Currently held units (default 0)

    Returns:
        Tuple of (units_to_trade, dollar_amount)
        - Positive units_to_trade = buy
        - Negative units_to_trade = sell
        - dollar_amount is always positive (absolute value)
    """
    if current_price <= 0:
        return 0.0, 0.0

    # Target allocation in dollars
    target_allocation_dollars = target_allocation_pct * portfolio_value

    # Current holding value
    current_holding_value = current_units * current_price

    # Rebalance needed
    rebalance_dollars = target_allocation_dollars - current_holding_value

    # Units to trade
    units_to_trade = rebalance_dollars / current_price

    # Dollar amount (absolute value)
    dollar_amount = abs(rebalance_dollars)

    return round(units_to_trade, 2), round(dollar_amount, 2)


def aggregate_portfolio_value(
    holdings: Dict[str, float], prices: Dict[str, float], cash: float
) -> float:
    """Calculate total portfolio value.

    Args:
        holdings: Dict of ticker -> units
        prices: Dict of ticker -> current price
        cash: Available cash

    Returns:
        Total portfolio value = sum(units × price) + cash
    """
    holdings_value = sum(
        units * prices.get(ticker, 0.0) for ticker, units in holdings.items()
    )
    return holdings_value + cash


def calculate_portfolio_weights(
    holdings: Dict[str, float], prices: Dict[str, float], cash: float
) -> Dict[str, float]:
    """Calculate current allocation % per ticker.

    Args:
        holdings: Dict of ticker -> units
        prices: Dict of ticker -> current price
        cash: Available cash

    Returns:
        Dict of ticker -> allocation percentage (0-100)
    """
    total_value = aggregate_portfolio_value(holdings, prices, cash)

    if total_value <= 0:
        return {}

    weights = {}
    for ticker, units in holdings.items():
        price = prices.get(ticker, 0.0)
        if price > 0:
            value = units * price
            pct = (value / total_value) * 100.0
            weights[ticker] = round(pct, 2)

    return weights


def estimate_risk_level(
    action: str, current_units: float, units_to_trade: float
) -> str:
    """Classify risk level based on action and position change.

    Args:
        action: Buy, Hold, or Sell
        current_units: Currently held units
        units_to_trade: Units to buy (positive) or sell (negative)

    Returns:
        "Low", "Medium", or "High"
    """
    # No change = low risk
    if abs(units_to_trade) < 0.01:
        return "Low"

    # Sell action = lower risk (trimming position)
    if action.lower() == "sell":
        return "Low"

    # Buying into new position = medium risk
    if current_units < 0.01 and units_to_trade > 0:
        return "Medium"

    # Large addition to existing position = higher risk
    if units_to_trade > 0 and current_units > 0:
        addition_pct = (units_to_trade / current_units) * 100.0
        if addition_pct > 50:
            return "High"
        elif addition_pct > 25:
            return "Medium"
        else:
            return "Low"

    # Default: medium
    return "Medium"
