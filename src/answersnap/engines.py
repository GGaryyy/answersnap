"""Which engines can be asked, and how: live through the adapters, or from fixtures.

Every live call goes through Provider.ask, so a snapshot is produced by exactly
the code path the recorded-response contract tests exercise. There is no
second, hand-rolled client anywhere in this package.
"""

import json
from datetime import datetime, timezone
from importlib import resources

from answersnap.providers import ProviderAnswer, RawCitation, get_provider_class

# ---------------------------------------------------------------- constants
ENGINE_LABELS = {"anthropic": "Claude", "openai": "ChatGPT", "google": "Gemini"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "google": "GOOGLE_API_KEY"}
# Rough per-call cost in USD, for the plan printed before a live run. Only
# Anthropic has a measurement (≈$0.165 including web search, n=1); the others
# print "no estimate" rather than a guess dressed as a number.
COST_PER_CALL_USD = {"anthropic": 0.17, "openai": None, "google": None}
FIXTURE_PACKAGE = "answersnap.examples"
FIXTURE_DIR = "fixtures"
NO_FIXTURE_TEXT = "[dry run] No fixture answer exists for this question."

# Skip reasons, recorded in the manifest and named in the report.
SKIPPED_NO_CREDENTIALS = "skipped_no_credentials"
SKIPPED_UNVERIFIED = "skipped_unverified"


def label(engine):
    return ENGINE_LABELS.get(engine, engine)


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
            stop_reason="fixture")


def live_status(engine):
    """(status, reason): "ok", or why this engine cannot be asked right now."""
    provider_class = get_provider_class(engine)
    if not provider_class.response_shape_verified():
        return SKIPPED_UNVERIFIED, (f"{label(engine)}: parser has no verified recorded "
                                    "response; it is not allowed to produce data")
    if not provider_class().credentials_available():
        return SKIPPED_NO_CREDENTIALS, f"{label(engine)}: not run — {KEY_ENV[engine]} is not set"
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
