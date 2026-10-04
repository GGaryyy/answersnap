"""Markdown rendering: the same two layers as the HTML, for diffs and pasting.

Answer text is untrusted. Many Markdown viewers render inline HTML and links,
so every value from an answer, a config or a URL goes through _inline.
"""

from answersnap.report.format import (
    engine_status_text,
    faith_row_label,
    faith_summary_text,
    metric_text,
    missing_metric_text,
    yes_blank,
)

# ---------------------------------------------------------------- constants
INLINE_ESCAPES = {"<": "&lt;", ">": "&gt;", "[": "\\[", "]": "\\]",
                  "*": "\\*", "`": "\\`", "|": "\\|"}


def _inline(text):
    """Escaped for inline Markdown: no raw HTML, no links, no emphasis."""
    return "".join(INLINE_ESCAPES.get(c, c) for c in (text or "")).replace("\n", " ")


def _cell(text):
    return _inline(text)


def _sentence(row):
    if not row["brand_segments"]:
        return ""
    return "".join(f"**{_inline(t)}**" if brand else _inline(t)
                   for t, brand in row["brand_segments"])


def _cited(row):
    if not row["citations_observable"]:
        return "not observable"
    return ", ".join(f"`{url.replace('`', '')}`" for url in row["owned_cited_urls"])


def _headline(report):
    lines = ["## Named in recommendation answers", ""]
    for item, section in zip(report["headline"]["per_engine"], report["engines"]):
        metric = item["named_in_rec_answers"]
        text = metric_text(metric) if metric else missing_metric_text(section)
        lines.append(f"- **{item['label']}**: {_inline(text)}")
    across = report["headline"]["across_engines"]
    if across["n"]:
        lines += ["", f"Across engines: named in **{across['count']} of {across['n']}** "
                      "recommendation answers. This is a count, not a rate — engines are "
                      "not averaged together."]
    return lines


def _engine_table(report):
    lines = ["", "## By engine", "",
             "| Engine | Model | Mention | Named in rec. answers | Citation | Faithfulness |",
             "|---|---|---|---|---|---|"]
    for section in report["engines"]:
        m = section["metrics"] or {}
        lines.append(" | ".join([
            f"| {section['label']}", _cell(engine_status_text(section)),
            metric_text(m.get("mention")), metric_text(m.get("named_in_rec_answers")),
            metric_text(m.get("citation")), _cell(faith_summary_text(section["faithfulness"]))])
            + " |")
    return lines


def _competitors(report):
    if not report["competitors"]:
        return []
    lines = ["", "## Competitor mentions", "",
             "| Engine | " + " | ".join(_cell(c) for c in report["competitors"]) + " |",
             "|---|" + "---|" * len(report["competitors"])]
    for section in report["engines"]:
        if not section["metrics"]:
            continue
        cells = [metric_text(c["mention"]) for c in section["metrics"]["competitors"]]
        lines.append(f"| {section['label']} | " + " | ".join(cells) + " |")
    return lines


def _evidence(report):
    lines = ["", "## Evidence", ""]
    for section in report["engines"]:
        rows = [r for r in report["rows"] if r["engine"] == section["engine"]]
        if not rows:
            continue
        lines += [f"### {section['label']}", "",
                  "| Question | Intent | Run | Sentence naming the brand | Mentioned | "
                  "Named in rec. answer | Owned citation | Faithfulness | File |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            faith = "; ".join(faith_row_label(f) for f in row["faithfulness"]) or "–"
            lines.append("| " + " | ".join([
                _cell(row["query_text"]), _cell(row["intent"]), str(row["repeat"] + 1),
                _sentence(row), yes_blank(row["mentioned"]),
                yes_blank(row["named_in_rec_answer"]), _cited(row), _cell(faith),
                f"`{row['file']}`"]) + " |")
        lines.append("")
    return lines


def _footer(report):
    run = report["run"]
    lines = ["## Definitions", ""]
    lines += [f"- **{d['term']}.** {d['text']}" for d in report["definitions"]]
    instrument = report["faithfulness_instrument"]
    lines += ["", "## Run", "",
              f"- Run `{run['run_id']}` · mode `{run['mode']}` · status `{run['status']}`",
              f"- Prompt set v{run['prompt_set']['version']} `{run['prompt_set']['hash']}` · "
              f"{run['prompt_set']['n_queries']} questions × {run['repeats']} runs",
              f"- Started {run['started_at']} · finished {run['finished_at']}",
              f"- answersnap {report['tool_version']} · faithfulness method "
              f"`{instrument['method_version']}` · self-check "
              f"{'passed' if instrument['passed'] else 'FAILED'}"]
    return lines


def render_markdown(report):
    lines = [f"# AI visibility snapshot: {_inline(report['brand']['name'])}", ""]
    lines += [f"> {_inline(notice)}" for notice in report["notices"]]
    if report["notices"]:
        lines.append("")
    lines += _headline(report) + _engine_table(report) + _competitors(report)
    lines += _evidence(report) + _footer(report)
    return "\n".join(lines) + "\n"
