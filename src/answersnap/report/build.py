"""Assemble the report from snapshots alone.

The report is a pure function of the run directory: the frozen config, the
manifest and the answer files. Editing your config afterwards cannot change a
report, and anyone holding the directory can rebuild it and get the same thing.
"""

from answersnap import __version__
from answersnap.engines import label
from answersnap.metrics import rates
from answersnap.metrics.excerpt import ELLIPSIS
from answersnap.metrics.stats import MIN_SAMPLES_FOR_RATE
from answersnap.metrics.usage import cost_totals, usage_totals
from answersnap.run import raw_status_on_disk

# ---------------------------------------------------------------- constants
REPORT_SCHEMA = "report-2"
SPAN_EXCERPT_CHARS = 160
SPANS_RECORDED, SPANS_NONE, SPANS_NOT_RECORDED = "recorded", "none", "not_recorded"
DEFINITIONS = (
    ("Mention",
     "Answers that name the brand (or any listed alias), out of all answers from "
     "that engine. Matching is deterministic string matching on your aliases; no "
     "model decides whether you were mentioned."),
    ("Named in recommendation answers",
     "Answers to questions whose intent is listed in recommendation_intents that "
     "name the brand. This is not the same as being recommended: an answer that "
     "names you to warn against you still counts. This version does not judge "
     "sentences; read the sentence column."),
    ("Citation",
     "Answers that cite a page on your owned domains, out of the answers where "
     "the engine exposed which sources it cited. Sources it retrieved but did not "
     "cite do not count. \"Not observable\" means the engine did not show its "
     "sources, which is not the same as citing nothing."),
    ("Faithfulness",
     "For engines that return the quoted text with each citation, whether that "
     "quote appears on the cited page. \"Not found verbatim\" is a lead to check, "
     "not a verdict: pages change, render with JavaScript and sit behind "
     "paywalls. Engines that return links without quotes are not observable."),
    ("Intervals",
     f"Every rate carries a 95% Wilson interval. Below {MIN_SAMPLES_FOR_RATE} "
     "answers a rate is marked low sample. Between two snapshots, a change "
     "smaller than these intervals is not a change."),
    ("Engines",
     "Engines are never averaged together; they are not samples of one "
     "population. The cross-engine line is a count with its denominator, not a "
     "rate."),
    ("Searched for",
     "The search queries the engine itself reported sending before it answered. "
     "\"Not searched\" means it answered without searching; \"not recorded\" "
     "means this snapshot predates query capture or the engine did not say."),
    ("Spans",
     "The parts of the answer the engine said a source supports, cut from the "
     "answer by the engine's own offsets. Claude marks whole passages, ChatGPT and "
     "Gemini mark ranges. A span the engine sent that did not land inside the "
     "answer text is dropped and counted, never moved to a guessed position."),
    ("Cost estimate",
     "Token usage as each engine reported it, priced at published list prices on "
     "the date shown. Not a bill: discounts, free allowances and price changes "
     "are not reflected, and models without a listed price get no estimate. "
     "Output includes reasoning for Claude and ChatGPT; Gemini reports it "
     "separately and bills it as output. Gemini reports no search count, so its "
     "search charges are not included. Never summed across engines."),
)


def _highlight_segments(sentence, highlights):
    """[(text, is_brand)] so renderers can mark the brand without re-matching."""
    if not sentence:
        return []
    segments, cursor = [], 0
    for start, end in highlights:
        if start > cursor:
            segments.append((sentence[cursor:start], False))
        segments.append((sentence[start:end], True))
        cursor = end
    if cursor < len(sentence):
        segments.append((sentence[cursor:], False))
    return segments


def _faithfulness_by_answer(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["engine"], row["query_index"], row["repeat"]), []).append(row)
    return grouped


def _span_excerpt(text, span):
    piece = text[span["start"]:span["end"]].strip()
    if len(piece) > SPAN_EXCERPT_CHARS:
        piece = piece[:SPAN_EXCERPT_CHARS].rstrip() + ELLIPSIS
    return {"start": span["start"], "end": span["end"], "text": piece}


def _spans_status(record):
    if record.get("spans") is None:
        return SPANS_NOT_RECORDED
    return SPANS_RECORDED if record["spans"] else SPANS_NONE


def cited_sources(record, owned_domains):
    """Each cited source with the parts of the answer it was said to support."""
    spans = record.get("spans") or []
    sources = []
    for index, citation in enumerate(record.get("citations") or []):
        if not citation.get("is_cited"):
            continue
        sources.append({
            "url": citation["url"], "domain": citation.get("domain"),
            "owned": rates.is_owned(citation.get("domain"), owned_domains),
            "excerpts": [_span_excerpt(record["answer_text"], span) for span in spans
                         if index in span["citation_indexes"]]})
    return sources


def _record_fields(record, owned_domains):
    sources = cited_sources(record, owned_domains)
    return {
        "searches": record.get("searches"),
        "search_count": record.get("search_count"),
        "search_ran": (record.get("params") or {}).get("search_ran"),
        "spans": record.get("spans"),
        "spans_dropped": record.get("spans_dropped"),
        "spans_status": _spans_status(record),
        "usage": record.get("usage"),
        "cost_estimate_usd": record.get("cost_estimate_usd"),
        "raw_file": record.get("raw_file"),
        "cited_sources": sources,
        "owned_cited_sources": [s for s in sources if s["owned"]],
    }


def evidence_rows(analysed, records, faith_rows, owned_domains=()):
    by_key = {(r["engine"], r["query_index"], r["repeat"]): r for r in records}
    faith = _faithfulness_by_answer(faith_rows)
    out = []
    for row in analysed:
        key = (row["engine"], row["query_index"], row["repeat"])
        record = by_key[key]
        out.append({
            **row,
            "id": f"row-{row['engine']}-q{row['query_index']:02d}-r{row['repeat']}",
            "query_text": record["query_text"],
            "intent": record["intent"],
            "model": record["model"],
            "observed_at": record["observed_at"],
            "file": record["_file"],
            "brand_segments": _highlight_segments(row["brand_sentence"],
                                                  row["brand_highlights"]),
            "faithfulness": faith.get(key, []),
            **_record_fields(record, owned_domains),
        })
    return out


def _engine_section(name, manifest_entry, metrics, faith_summary, records):
    mine = [r for r in records if r["engine"] == name and not rates.is_placeholder(r)]
    return {"engine": name, "label": label(name),
            "usage": usage_totals(mine), "cost": cost_totals(mine),
            "status": manifest_entry["status"], "reason": manifest_entry.get("reason"),
            "models": manifest_entry.get("models_reported") or [],
            "planned": manifest_entry.get("planned", 0), "done": manifest_entry.get("done", 0),
            "failed": manifest_entry.get("failed", 0),
            "metrics": metrics, "faithfulness": faith_summary}


def _notices(manifest, config_warnings):
    notices = list(config_warnings)
    if manifest["mode"] == "dry_run":
        notices.insert(0, "DRY RUN: answers come from bundled fictional fixtures, not "
                          "from any engine. Nothing here describes a real brand.")
    engines = manifest["engines"]
    if manifest["status"] == "incomplete":
        done = sum(e.get("done", 0) for e in engines.values())
        planned = sum(e.get("planned", 0) for e in engines.values())
        notices.append(f"Incomplete run: {done} of {planned} planned answers were collected; "
                       f"{manifest['calls']['failed']} calls failed.")
    not_asked = [label(name) for name, e in engines.items() if e["status"] == "skipped_not_requested"]
    if not_asked:
        notices.append(f"Not asked in this run: {', '.join(not_asked)}. The report says nothing "
                       "about how those engines answer.")
    notices.extend(entry["reason"] for entry in engines.values() if entry.get("reason"))
    return notices


def build_report(manifest, config, records, faith_rows, faith_summary, instrument,
                 *, generated_at, errors=()):
    analysed, per_engine, across = rates.compute(records, config)
    engines = [_engine_section(name, manifest["engines"][name], per_engine.get(name),
                               faith_summary.get(name), records)
               for name in config.engines]
    return {
        "schema": REPORT_SCHEMA,
        "tool_version": __version__,
        "generated_at": generated_at.isoformat(),
        "run": {key: manifest[key] for key in
                ("run_id", "mode", "status", "started_at", "finished_at", "repeats",
                 "prompt_set", "config_hash", "calls", "tool_version")}
               | {"record_schemas_on_disk": sorted({r.get("schema") for r in records
                                                    if r.get("schema")}),
                  # From the records: a resume may have changed --no-raw, and
                  # runs from before raw capture say nothing.
                  "raw_payloads": raw_status_on_disk(records)},
        "brand": {"name": config.brand.name, "aliases": list(config.brand.aliases),
                  "owned_domains": list(config.brand.owned_domains)},
        "competitors": [c.name for c in config.competitors],
        "recommendation_intents": list(config.recommendation_intents),
        "queries": [{"index": i, "text": q.text, "intent": q.intent}
                    for i, q in enumerate(config.prompt_set.queries)],
        "headline": {"per_engine": [{"engine": e["engine"], "label": e["label"],
                                     "status": e["status"],
                                     "named_in_rec_answers": (e["metrics"] or {}).get(
                                         "named_in_rec_answers")}
                                    for e in engines],
                     "across_engines": across},
        "engines": engines,
        "rows": evidence_rows(analysed, records, faith_rows, config.brand.owned_domains),
        "errors": list(errors),
        "faithfulness_instrument": instrument,
        "notices": _notices(manifest, config.warnings()),
        "definitions": [{"term": t, "text": d} for t, d in DEFINITIONS],
    }
