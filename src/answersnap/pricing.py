"""Estimated cost from recorded token usage at published list prices.

An estimate, never a bill: list prices change, accounts get discounts and free
allowances, and an answer whose model has no row here gets no number at all
rather than a guess. Every estimate carries the date the prices were read, so
a report can always say what it was priced against.

Self-contained on purpose: deleting this module (and its one call site in
run.py) leaves every snapshot intact.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------- constants
PRICES_AS_OF = "2026-10-05"
PRICE_PAGES = {
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "openai": "https://developers.openai.com/api/docs/pricing",
    "google": "https://ai.google.dev/gemini-api/docs/pricing",
}
PER_MILLION = 1_000_000
# A reported model id may carry a dated snapshot suffix (gpt-5.1-2025-11-13).
SNAPSHOT_SUFFIX = "-20"


@dataclass(frozen=True)
class ListPrice:
    input_per_million: float
    output_per_million: float
    # None where search is not billed per request we can count.
    search_per_request: float | None = None


# Standard tier, global routing, no batch, no caching (answersnap requests
# never set cache_control). Search content tokens arrive inside input_tokens.
LIST_PRICES = {
    # $2 / $10 per MTok; web search $10 per 1,000 searches.
    ("anthropic", "claude-sonnet-5"): ListPrice(2.00, 10.00, 0.01),
    # $1.25 / $10 per MTok; web search $10 per 1,000 calls for the gpt-5 family.
    ("openai", "gpt-5.1"): ListPrice(1.25, 10.00, 0.01),
    # $0.75 / $3.75 per MTok through 2026-12-31 (output includes thinking).
    # Grounding is $14 per 1,000 requests after 5,000 free a month, and the API
    # reports no request count, so search is left unpriced.
    ("google", "gemini-3.7-flash"): ListPrice(0.75, 3.75, None),
}

REASON_NO_PRICE = "no_list_price"
REASON_NO_USAGE = "no_usage"
REASON_DRY_RUN = "dry_run"


def list_price(engine, model):
    if not model:
        return None, None
    for (price_engine, price_model), price in LIST_PRICES.items():
        if price_engine != engine:
            continue
        if model == price_model or model.startswith(price_model + SNAPSHOT_SUFFIX):
            return price_model, price
    return None, None


def no_estimate(reason):
    return None, {"kind": "none", "reason": reason}


def _billed_output(usage):
    output = usage.get("output_tokens")
    if output is None:
        return None
    if usage.get("output_includes_reasoning"):
        return output
    return output + (usage.get("reasoning_tokens") or 0)


def estimate_cost_usd(engine, model, usage, search_count):
    """(usd or None, basis). basis says what the number is, or why there is none."""
    priced_model, price = list_price(engine, model)
    if price is None:
        return no_estimate(REASON_NO_PRICE)
    if not usage or usage.get("input_tokens") is None or _billed_output(usage) is None:
        return no_estimate(REASON_NO_USAGE)
    usd = (usage["input_tokens"] * price.input_per_million
           + _billed_output(usage) * price.output_per_million) / PER_MILLION
    searches_priced = price.search_per_request is not None and search_count is not None
    if searches_priced:
        usd += search_count * price.search_per_request
    return round(usd, 6), {"kind": "list_price", "as_of": PRICES_AS_OF,
                           "model": priced_model, "searches_priced": searches_priced}


def basis_label(dates=None):
    return f"list price as of {', '.join(dates or [PRICES_AS_OF])}"
