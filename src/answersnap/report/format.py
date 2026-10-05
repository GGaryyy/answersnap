"""Wording shared by every renderer, so Markdown and HTML can never disagree."""

from answersnap.metrics import rates

# ---------------------------------------------------------------- constants
STATUS_WORDS = {
    rates.STATUS_NOT_OBSERVABLE: "not observable",
    rates.STATUS_NOT_MEASURED: "not measured",
    rates.STATUS_NO_DATA: "no answers",
}
FAITH_STATUS_WORDS = {
    "not_observable": "not observable: engine does not return quoted text",
    "not_checked": "not checked (fetching was switched off)",
    "not_checkable": "not checkable: none of the quoted pages could be read",
    "no_citations": "no cited sources",
}
NO_MEASURABLE_ANSWERS = "no measurable answers"
LOW_SAMPLE_NOTE = "low sample"
NOT_CHECKABLE_WORDS = {
    "no_cited_text": "no quote returned",
    "not_fetched": "not checked (fetching was switched off)",
    "fetch_not_in_dry_run": "page not bundled with the dry run",
    "fetch_robots_blocked": "not checkable: robots.txt disallows fetching",
    "fetch_blocked_unsafe": "not checkable: address refused by the fetch guard",
    "fetch_paywalled": "not checkable: paywalled",
    "fetch_auth_required": "not checkable: login required",
    "fetch_gone": "not checkable: page is gone (404/410)",
    "fetch_timeout": "not checkable: timed out",
    "fetch_http_error": "not checkable: HTTP error",
    "no_page_text": "not checkable: page has no readable text",
}


def pct(value):
    return "–" if value is None else f"{value * 100:.0f}%"


def has_value(metric):
    return bool(metric) and metric["status"] in (rates.STATUS_OK, rates.STATUS_LOW_SAMPLE)


def interval(metric):
    return f"95% CI {pct(metric['ci_low'])}–{pct(metric['ci_high'])}"


def metric_text(metric):
    """"4 of 9 (44%, 95% CI 19–73%)", or the reason there is no number."""
    if not metric:
        return "no answers"
    if not has_value(metric):
        return STATUS_WORDS.get(metric["status"], metric["status"])
    text = f"{metric['count']} of {metric['n']} ({pct(metric['rate'])}, {interval(metric)})"
    if metric["status"] == rates.STATUS_LOW_SAMPLE:
        text += f" · {LOW_SAMPLE_NOTE}"
    return text


def faith_summary_text(summary):
    if not summary:
        return "no cited sources"
    if summary["status"] != "ok":
        return FAITH_STATUS_WORDS.get(summary["status"], summary["status"])
    text = (f"{summary['found']} of {summary['checked']} checked quotes found on the page "
            f"verbatim; {summary['not_found']} not found")
    if summary["not_checkable"]:
        text += f"; {summary['not_checkable']} more could not be checked"
    return text


def faith_row_text(row):
    if row["result"] == "found":
        if row.get("matched_by") == "pieces":
            return f"found verbatim, in {row['pieces']} separate pieces"
        return "found verbatim"
    if row["result"] == "not_found":
        return f"not found verbatim (best overlap {row['best_window_score']:.2f})"
    reason = row["not_comparable_reason"] or ""
    return NOT_CHECKABLE_WORDS.get(reason, "not checkable: " + reason.replace("_", " "))


def faith_row_label(row):
    """One citation's result, named by its site, so a cell with several
    citations says which one was found and which was not."""
    return f"{row.get('domain') or 'unknown site'}: {faith_row_text(row)}"


def yes_blank(value):
    """True -> "yes", False -> "" (an empty cell is a result), None -> "–"."""
    if value is None:
        return "–"
    return "yes" if value else ""


def engine_status_text(section):
    if section["status"] in ("ok", "partial"):
        models = ", ".join(section["models"]) or "unknown model"
        if section["status"] == "ok":
            return models
        return f"{models} · partial: {section['done']} of {section['planned']}"
    return section.get("reason") or section["status"].replace("_", " ")


def missing_metric_text(section):
    """Why an engine has no number: skipped, or answered nothing measurable
    (a dry run of questions without fixtures)."""
    if section["status"] in ("ok", "partial"):
        return NO_MEASURABLE_ANSWERS
    return engine_status_text(section)


def searches_text(row):
    """What the engine searched for, or why there is nothing to show."""
    searches = row.get("searches")
    if searches is None:
        return "not recorded"
    if row.get("search_ran") is False and not searches:
        return "not searched"
    if not searches:
        return "no query recorded"
    text = " · ".join(s["query"] for s in searches)
    count = row.get("search_count")
    if count is not None and count != len(searches):
        text += f" ({count} searches reported)"
    return text


def _tokens(usage):
    text = f"{usage['input_tokens']:,} in" if usage["input_tokens"] is not None else ""
    if usage["output_tokens"] is not None:
        text += f" · {usage['output_tokens']:,} out"
    if usage["reasoning_tokens"]:
        text += f" · {usage['reasoning_tokens']:,} reasoning"
    return text.strip(" ·")


def cost_text(cost, dry_run):
    if dry_run:
        return "no cost (dry run)"
    if cost["usd"] is None:
        return "no estimate"
    text = f"≈ ${cost['usd']:.2f} ({cost['basis']})"
    if cost["answers_unpriced"]:
        text += f", {cost['answers_unpriced']} answers not priced"
    return text


def usage_text(section, dry_run):
    """One engine's tokens, searches and estimated cost. Never across engines."""
    usage, cost = section.get("usage"), section.get("cost")
    if not usage or not usage["with_usage"]:
        return "usage not recorded"
    parts = [_tokens(usage)]
    if usage["search_count"] is not None:
        parts.append(f"{usage['search_count']:,} searches")
    parts.append(cost_text(cost, dry_run))
    text = " · ".join(p for p in parts if p)
    if usage["with_usage"] < usage["answers"]:
        text = f"tokens recorded for {usage['with_usage']} of {usage['answers']} answers: " + text
    return text


def spans_note(row):
    """Shown when a row's cited sources cannot carry sentence excerpts."""
    if row["spans_status"] == "not_recorded":
        return "spans not recorded"
    if row.get("spans_dropped"):
        return f"{row['spans_dropped']} spans could not be placed"
    return ""
