"""Token usage and estimated cost totals, one engine at a time.

Never summed across engines: tokens are counted by each platform's own
tokenizer and priced at its own rates, so a run-wide total would add unlike
units. Sums cover only answers that reported usage, and every total says how
many answers that was — a partial sum is never presented as a total.
"""

from answersnap import pricing


def _token_sum(records, key):
    values = [r["usage"].get(key) for r in records if r.get("usage")]
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def usage_totals(records):
    """One engine's recorded usage. Sums cover only answers that reported it,
    and the counts say how many that was — a partial sum is never a total."""
    with_count = [r["search_count"] for r in records if r.get("search_count") is not None]
    return {"answers": len(records),
            "with_usage": sum(1 for r in records if r.get("usage")),
            "input_tokens": _token_sum(records, "input_tokens"),
            "output_tokens": _token_sum(records, "output_tokens"),
            "reasoning_tokens": _token_sum(records, "reasoning_tokens"),
            "search_count": sum(with_count) if with_count else None,
            "answers_with_search_count": len(with_count)}


def cost_totals(records):
    priced = [r["cost_estimate_usd"] for r in records if r.get("cost_estimate_usd") is not None]
    return {"usd": round(sum(priced), 4) if priced else None,
            "answers_priced": len(priced), "answers_unpriced": len(records) - len(priced),
            "basis": _basis(records) if priced else None}


def _basis(records):
    """Named from the dates the answers were priced at, not today's table, so a
    rebuilt report says what its numbers were priced against."""
    dates = sorted({(r.get("cost_basis") or {}).get("as_of") for r in records
                    if r.get("cost_estimate_usd") is not None} - {None})
    return pricing.basis_label(dates)
