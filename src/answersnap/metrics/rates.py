"""Per-engine rates over snapshot records. Statistics only — no verdicts.

Three rules run through everything here:

  - Never merge engines. They are not samples of one population, so an average
    across them describes none of them.
  - A rate that could not be measured has a status saying why, never 0.
  - Every rate carries its count, its denominator and a 95% interval, so a
    reader can see how little a small sample says.

"Named in recommendation answers" is deliberately not called "recommended":
it counts answers to recommendation-intent questions that name the brand, and
an answer that names a brand to warn against it still counts. Sentence-level
recommendation is a later, separate measurement.
"""

from answersnap.metrics.excerpt import answer_excerpt, brand_sentence
from answersnap.metrics.mention import extract_entities
from answersnap.metrics.stats import MIN_SAMPLES_FOR_RATE, wilson_interval
from answersnap.providers.citation_host import strip_www

# ---------------------------------------------------------------- constants
STATUS_OK = "ok"
STATUS_LOW_SAMPLE = "low_sample"
STATUS_NOT_OBSERVABLE = "not_observable"
STATUS_NOT_MEASURED = "not_measured"
STATUS_NO_DATA = "no_data"
ROLE_BRAND = "brand"
ROLE_COMPETITOR = "competitor"


def entity_table(config):
    brand = config.brand
    return ([(brand.name, ROLE_BRAND, list(brand.aliases))]
            + [(c.name, ROLE_COMPETITOR, list(c.aliases)) for c in config.competitors])


def is_placeholder(record):
    """A dry-run stand-in for a question with no fixture: not an answer."""
    return bool((record.get("params") or {}).get("no_fixture"))


def _owned(domain, owned_domains):
    host = strip_www((domain or "").lower())
    return any(host == owned or host.endswith("." + owned) for owned in owned_domains)


def _citation_view(record, owned_domains):
    """(observable, owned cited URLs). Only CITED sources count; a source the
    engine merely retrieved did not shape the answer."""
    if not record.get("cited_sources_available"):
        return False, []
    owned = [c["url"] for c in record.get("citations", [])
             if c.get("is_cited") and _owned(c.get("domain"), owned_domains)]
    return True, owned


def analyse_record(record, config):
    """Everything the evidence table shows for one answer."""
    text = record["answer_text"]
    mentions = extract_entities(text, entity_table(config))
    brand = next((m for m in mentions if m["entity_role"] == ROLE_BRAND), None)
    spans = [(s, e) for s, e, _ in brand["spans"]] if brand else []
    sentence, highlights = brand_sentence(text, spans)
    is_rec_question = record["intent"] in config.recommendation_intents
    observable, owned = _citation_view(record, config.brand.owned_domains)
    return {
        "engine": record["engine"],
        "query_index": record["query_index"],
        "repeat": record["repeat"],
        "mentioned": brand is not None,
        "brand_rank": brand["rank"] if brand else None,
        "brand_sentence": sentence,
        "brand_highlights": highlights,
        "excerpt": answer_excerpt(text),
        "competitors_named": [m["entity"] for m in mentions
                              if m["entity_role"] == ROLE_COMPETITOR],
        # None, not False, where the question is not a recommendation question:
        # "not asked" and "not named" are different facts.
        "named_in_rec_answer": (brand is not None) if is_rec_question else None,
        "citations_observable": observable,
        "owned_cited": (bool(owned) if config.brand.owned_domains else None)
                       if observable else None,
        "owned_cited_urls": owned,
    }


def metric(count, n, *, status_if_empty=STATUS_NO_DATA):
    if n == 0:
        return {"count": None, "n": 0, "rate": None, "ci_low": None, "ci_high": None,
                "status": status_if_empty}
    low, high = wilson_interval(count, n)
    return {"count": count, "n": n, "rate": count / n, "ci_low": low, "ci_high": high,
            "status": STATUS_OK if n >= MIN_SAMPLES_FOR_RATE else STATUS_LOW_SAMPLE}


def not_measured():
    return metric(0, 0, status_if_empty=STATUS_NOT_MEASURED)


def _count(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return sum(1 for v in values if v), len(values)


def engine_metrics(engine, rows, config):
    mention = metric(*_count(rows, "mentioned"))
    if config.recommendation_queries():
        named = metric(*_count(rows, "named_in_rec_answer"))
    else:
        named = not_measured()
    if not config.brand.owned_domains:
        citation = not_measured()
    else:
        citation = metric(*_count(rows, "owned_cited"), status_if_empty=STATUS_NOT_OBSERVABLE)
    competitors = [{"name": c.name,
                    "mention": metric(sum(1 for r in rows if c.name in r["competitors_named"]),
                                      len(rows))}
                   for c in config.competitors]
    return {"engine": engine, "answers": len(rows), "mention": mention,
            "named_in_rec_answers": named, "citation": citation, "competitors": competitors}


def compute(records, config):
    """(rows, per-engine metrics, cross-engine count). Placeholders excluded."""
    real = [r for r in records if not is_placeholder(r)]
    rows = sorted((analyse_record(r, config) for r in real),
                  key=lambda r: (r["engine"], r["query_index"], r["repeat"]))
    by_engine = {}
    for row in rows:
        by_engine.setdefault(row["engine"], []).append(row)
    per_engine = {engine: engine_metrics(engine, by_engine.get(engine, []), config)
                  for engine in config.engines if engine in by_engine}
    return rows, per_engine, cross_engine_count(per_engine)


def cross_engine_count(per_engine):
    """A COUNT across engines, not a rate: the engines are not one population,
    so this is "how many answers named you", with its denominator, and no %."""
    named = [m["named_in_rec_answers"] for m in per_engine.values()
             if m["named_in_rec_answers"]["n"]]
    return {"count": sum(m["count"] for m in named), "n": sum(m["n"] for m in named),
            "engines": [e for e, m in per_engine.items() if m["named_in_rec_answers"]["n"]]}
