"""Small builders for synthetic configs and answer records. Invented content only."""

from datetime import datetime, timezone

from answersnap.config import parse_config

STAMP = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

BASE_CONFIG = {
    "schema": 1,
    "brand": {"name": "Trellis", "aliases": ["Trellis Dental"], "owned_domains": ["trellis.example"]},
    "competitors": [{"name": "Kanbanly", "aliases": []}],
    "engines": ["anthropic", "openai"],
    "repeats": 2,
    "recommendation_intents": ["recommendation"],
    "prompt_set": {"version": 1, "queries": [
        {"text": "Best clinic for implants?", "intent": "recommendation"},
        {"text": "How long do implants last?", "intent": "how_to"},
    ]},
}


def config_dict(**overrides):
    document = {**BASE_CONFIG, **overrides}
    return document


def make_config(**overrides):
    return parse_config(config_dict(**overrides))


def citation(url, *, domain=None, is_cited=True, cited_text=None):
    host = domain or url.split("/")[2]
    return {"url": url, "title": None, "domain": host, "is_cited": is_cited,
            "position": 0 if is_cited else None, "cited_text": cited_text}


def record(engine="anthropic", query_index=0, repeat=0, text="Trellis is great.",
           intent="recommendation", citations=(), observable=True, params=None):
    return {"schema": "answer-1", "run_id": "r", "engine": engine, "model": "m",
            "params": params or {}, "query_index": query_index,
            "query_text": f"question {query_index}", "intent": intent, "repeat": repeat,
            "requested_at": STAMP.isoformat(), "observed_at": STAMP.isoformat(),
            "answer_text": text, "citations": list(citations),
            "cited_sources_available": observable, "retrieved_set_available": False,
            "stop_reason": "end_turn", "prompt_set_version": 1, "prompt_set_hash": "h",
            "_file": f"answers/{engine}/q{query_index:02d}_r{repeat}.json"}
