"""Token pricing, used to turn usage into an estimated cost.

Prices are published by the provider and change. They are listed here with the
date they were checked, and an unknown model yields no estimate at all rather
than a number that looks authoritative and is wrong.
"""

from __future__ import annotations

# USD per million tokens, as (input, output). Checked 30 August 2026 against
# the OpenAI pricing page. Verify before quoting these figures anywhere.
MODEL_PRICING_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

_PER_MILLION = 1_000_000


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate the cost of one run, or None when the model has no listed price."""
    price = MODEL_PRICING_USD_PER_MILLION.get(model)
    if price is None:
        return None

    input_price, output_price = price
    return (input_tokens * input_price + output_tokens * output_price) / _PER_MILLION
