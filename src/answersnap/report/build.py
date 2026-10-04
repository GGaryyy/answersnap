"""Assemble the report from snapshots alone.

The report is a pure function of the run directory: the frozen config, the
manifest and the answer files. Editing your config afterwards cannot change a
report, and anyone holding the directory can rebuild it and get the same thing.
"""

from answersnap import __version__
from answersnap.engines import label
from answersnap.metrics import rates
from answersnap.metrics.stats import MIN_SAMPLES_FOR_RATE

# ---------------------------------------------------------------- constants
REPORT_SCHEMA = "report-1"
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


def evidence_rows(analysed, records, faith_rows):
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
        })
    return out


def _engine_section(name, manifest_entry, metrics, faith_summary):
    return {"engine": name, "label": label(name),
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
                               faith_summary.get(name))
               for name in config.engines]
    return {
        "schema": REPORT_SCHEMA,
        "tool_version": __version__,
        "generated_at": generated_at.isoformat(),
        "run": {key: manifest[key] for key in
                ("run_id", "mode", "status", "started_at", "finished_at", "repeats",
                 "prompt_set", "config_hash", "calls", "tool_version")},
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
        "rows": evidence_rows(analysed, records, faith_rows),
        "errors": list(errors),
        "faithfulness_instrument": instrument,
        "notices": _notices(manifest, config.warnings()),
        "definitions": [{"term": t, "text": d} for t, d in DEFINITIONS],
    }
