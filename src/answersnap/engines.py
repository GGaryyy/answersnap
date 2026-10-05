"""Which engines can be asked, and how: live through the adapters, or from fixtures.

Every live call goes through Provider.ask, so a snapshot is produced by exactly
the code path the recorded-response contract tests exercise. There is no
second, hand-rolled client anywhere in this package.
"""

import json
import os
from datetime import datetime, timezone
from importlib import resources

from answersnap import pricing
from answersnap.providers import (
    AnswerSpan,
    ProviderAnswer,
    RawCitation,
    SearchQuery,
    get_provider_class,
)
from answersnap.providers.spans import usage_summary

# ---------------------------------------------------------------- constants
ENGINE_LABELS = {"anthropic": "Claude", "openai": "ChatGPT", "google": "Gemini"}
# The first name is the one to set; later ones are accepted aliases.
KEY_ENV = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",),
           "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
# Usage of one real call per engine (the recorded responses under
# providers/recorded/, n=1 each), priced at list price for the plan printed
# before a live run. A rough guide only: answers vary in length and in how
# many searches they make. After a run, cost comes from each answer's own usage.
TYPICAL_USAGE = {
    "anthropic": {"model": "claude-sonnet-5", "search_count": 2,
                  "usage": {"input_tokens": 17599, "output_tokens": 1949,
                            "reasoning_tokens": 262, "output_includes_reasoning": True}},
    "openai": {"model": "gpt-5.1", "search_count": 1,
               "usage": {"input_tokens": 7464, "output_tokens": 1700,
                         "reasoning_tokens": 57, "output_includes_reasoning": True}},
    "google": {"model": "gemini-3.7-flash", "search_count": None,
               "usage": {"input_tokens": 211, "output_tokens": 1448,
                         "reasoning_tokens": 817, "output_includes_reasoning": False}},
}
FIXTURE_PACKAGE = "answersnap.examples"
FIXTURE_DIR = "fixtures"
NO_FIXTURE_TEXT = "[dry run] No fixture answer exists for this question."

# Skip reasons, recorded in the manifest and named in the report.
SKIPPED_NO_CREDENTIALS = "skipped_no_credentials"
SKIPPED_UNVERIFIED = "skipped_unverified"


def label(engine):
    return ENGINE_LABELS.get(engine, engine)


def per_call_estimate_usd(engine, model=None):
    """List-price cost of one typical call, or None when it cannot be priced."""
    typical = TYPICAL_USAGE.get(engine)
    if typical is None:
        return None
    usd, _ = pricing.estimate_cost_usd(engine, model or typical["model"],
                                       typical["usage"], typical["search_count"])
    return usd


def key_names(engine):
    return " or ".join(KEY_ENV[engine])


def key_present(engine):
    return any(os.environ.get(name) for name in KEY_ENV[engine])


class LiveEngine:
    mode = "live"

    def __init__(self, name, provider):
        self.name = name
        self._provider = provider

    @property
    def model(self):
        return self._provider.model

    def answer(self, query_text, repeat):
        # The repeat index is ours, not the engine's: every call is a fresh
        # conversation with the query sent verbatim.
        return self._provider.ask(query_text)


def _fixture_document(engine):
    path = resources.files(FIXTURE_PACKAGE) / FIXTURE_DIR / f"{engine}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_citations(entries):
    # Position is the order among CITED sources, as the adapters record it;
    # retrieved-only sources carry none.
    citations, cited_count = [], 0
    for entry in entries:
        is_cited = entry.get("is_cited", True)
        citations.append(RawCitation(
            url=entry["url"], title=entry.get("title"),
            position=cited_count if is_cited else None, is_cited=is_cited,
            cited_text=entry.get("cited_text"),
            resolved_domain=entry.get("resolved_domain")))
        cited_count += is_cited
    return citations


def _fixture_searches(variant):
    searches = variant.get("searches")
    if searches is None:
        return None
    return [SearchQuery(query=s["query"], results_count=s.get("results_count"))
            for s in searches]


def _fixture_spans(variant):
    spans = variant.get("spans")
    if spans is None:
        return None
    return [AnswerSpan(start=s["start"], end=s["end"],
                       citation_indexes=tuple(s["citation_indexes"])) for s in spans]


def _fixture_usage(variant):
    usage = variant.get("usage")
    if usage is None:
        return None
    return usage_summary(usage.get("input_tokens"), usage.get("output_tokens"),
                         usage.get("reasoning_tokens"), usage.get("total_tokens"),
                         usage.get("output_includes_reasoning", True), usage)


class FixtureEngine:
    """Answers from bundled fictional fixtures: zero network, zero cost.

    Questions without a fixture still get an answer — a labelled placeholder
    with citations marked unobservable — so a dry run of your own config shows
    the full report shape instead of failing.
    """

    mode = "dry_run"

    def __init__(self, name, clock=None):
        self.name = name
        self._document = _fixture_document(name)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def model(self):
        return self._document["model"]

    def _params(self):
        return {"model_requested": self.model, "model_reported_by_platform": True,
                "fixture": True}

    def _search_count(self, variant):
        # Mirrors the platform: Gemini reports queries but no billed count.
        if variant.get("searches") is None or not self._document.get("search_count_reported", True):
            return None
        return len(variant["searches"])

    def answer(self, query_text, repeat):
        stamp = self._clock()
        variants = self._document["answers"].get(query_text)
        if not variants:
            # Marked so metrics leave it out of every denominator: it is not an
            # answer, and counting it would print a fabricated 0%.
            return ProviderAnswer(platform=self.name, model=self.model,
                                  params={**self._params(), "no_fixture": True},
                                  requested_at=stamp, observed_at=stamp,
                                  answer_text=NO_FIXTURE_TEXT, citations=[],
                                  cited_sources_available=False, stop_reason="fixture")
        variant = variants[repeat % len(variants)]
        return ProviderAnswer(
            platform=self.name, model=self.model, params=self._params(),
            requested_at=stamp, observed_at=stamp, answer_text=variant["answer_text"],
            citations=_fixture_citations(variant.get("citations", [])),
            cited_sources_available=self._document["cited_sources_available"],
            retrieved_set_available=self._document["retrieved_set_available"],
            stop_reason="fixture",
            searches=_fixture_searches(variant),
            search_count=self._search_count(variant),
            spans=_fixture_spans(variant),
            usage=_fixture_usage(variant))


def live_status(engine):
    """(status, reason): "ok", or why this engine cannot be asked right now."""
    provider_class = get_provider_class(engine)
    if not provider_class.response_shape_verified():
        return SKIPPED_UNVERIFIED, (f"{label(engine)}: parser has no verified recorded "
                                    "response; it is not allowed to produce data")
    if not provider_class().credentials_available():
        return SKIPPED_NO_CREDENTIALS, f"{label(engine)}: not run — {key_names(engine)} is not set"
    return "ok", None


def build_engines(names, dry_run=False, clock=None):
    """Return (engines, skipped). skipped maps name -> (status, reason)."""
    engines, skipped = {}, {}
    for name in names:
        if dry_run:
            engines[name] = FixtureEngine(name, clock=clock)
            continue
        status, reason = live_status(name)
        if status != "ok":
            skipped[name] = (status, reason)
            continue
        engines[name] = LiveEngine(name, get_provider_class(name)())
    return engines, skipped
