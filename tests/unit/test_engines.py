import pytest
from builders import STAMP

from answersnap import engines
from answersnap.config import SUPPORTED_ENGINES
from answersnap.examples import __name__ as _examples  # noqa: F401  (package must import)
from answersnap.providers import get_provider_class

EXAMPLE_QUESTION = "What's the best coffee subscription for someone getting into espresso?"


@pytest.mark.parametrize("engine", SUPPORTED_ENGINES)
def test_every_supported_engine_has_a_verified_recorded_response(engine):
    assert get_provider_class(engine).response_shape_verified()


@pytest.mark.parametrize("engine", SUPPORTED_ENGINES)
def test_fixture_engine_answers_example_questions_without_any_client(engine, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a fixture answer must not build a client")
    monkeypatch.setattr(get_provider_class(engine), "ask", refuse)
    fixture = engines.FixtureEngine(engine, clock=lambda: STAMP)
    answer = fixture.answer(EXAMPLE_QUESTION, 0)
    assert answer.platform == engine
    assert answer.answer_text and answer.observed_at == STAMP
    assert any(c.is_cited for c in answer.citations)


def test_fixture_repeats_differ_and_wrap():
    fixture = engines.FixtureEngine("anthropic")
    texts = [fixture.answer(EXAMPLE_QUESTION, r).answer_text for r in range(4)]
    assert len(set(texts[:3])) == 3
    assert texts[3] == texts[0]


def test_fixture_positions_number_cited_sources_only():
    answer = engines.FixtureEngine("anthropic").answer(EXAMPLE_QUESTION, 0)
    assert [c.position for c in answer.citations] == [0, 1, None]


def test_unknown_question_gets_a_marked_placeholder():
    answer = engines.FixtureEngine("openai").answer("Something not in the fixtures?", 0)
    assert answer.answer_text == engines.NO_FIXTURE_TEXT
    assert answer.params["no_fixture"] is True
    assert answer.cited_sources_available is False


def test_only_anthropic_fixtures_carry_quotes():
    for engine, expect_quotes in (("anthropic", True), ("openai", False), ("google", False)):
        answer = engines.FixtureEngine(engine).answer(EXAMPLE_QUESTION, 0)
        has_quotes = any(c.cited_text for c in answer.citations)
        assert has_quotes is expect_quotes, engine


def test_google_fixture_keeps_the_resolved_domain():
    answer = engines.FixtureEngine("google").answer(EXAMPLE_QUESTION, 0)
    assert answer.citations[0].url.startswith("https://vertexaisearch")
    assert answer.citations[0].domain == "sample-roasters.example"


@pytest.mark.parametrize("engine", SUPPORTED_ENGINES)
def test_missing_key_is_a_named_skip(engine, monkeypatch):
    for name in engines.KEY_ENV[engine]:
        monkeypatch.delenv(name, raising=False)
    status, reason = engines.live_status(engine)
    assert status == engines.SKIPPED_NO_CREDENTIALS
    assert engines.KEY_ENV[engine][0] in reason


@pytest.mark.parametrize("name", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_gemini_key_is_found_under_either_name(name, monkeypatch):
    for other in engines.KEY_ENV["google"]:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "test-key-not-real")
    assert engines.key_present("google")
    assert engines.live_status("google") == ("ok", None)


def test_the_documented_gemini_name_wins_when_both_are_set(monkeypatch):
    from answersnap.providers.google_provider import GoogleProvider
    monkeypatch.setenv("GEMINI_API_KEY", "preferred")
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback")
    assert GoogleProvider()._api_key == "preferred"


def test_build_engines_skips_unavailable_and_keeps_order(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    live, skipped = engines.build_engines(["openai", "anthropic"])
    assert list(live) == ["anthropic"]
    assert skipped["openai"][0] == engines.SKIPPED_NO_CREDENTIALS
    dry, none_skipped = engines.build_engines(["openai", "anthropic"], dry_run=True)
    assert list(dry) == ["openai", "anthropic"] and none_skipped == {}


def test_unverified_parser_is_refused_before_credentials(monkeypatch):
    cls = get_provider_class("anthropic")
    monkeypatch.setattr(cls, "response_shape_verified", classmethod(lambda c: False))
    status, reason = engines.live_status("anthropic")
    assert status == engines.SKIPPED_UNVERIFIED
    assert "not allowed to produce data" in reason


@pytest.mark.parametrize("engine", SUPPORTED_ENGINES)
def test_every_fixture_span_cuts_real_text_and_points_at_a_cited_source(engine):
    fixture = engines.FixtureEngine(engine)
    for question, variants in fixture._document["answers"].items():
        for repeat in range(len(variants)):
            answer = fixture.answer(question, repeat)
            assert answer.searches is not None and answer.usage is not None
            for span in answer.spans:
                assert answer.answer_text[span.start:span.end].strip()
                assert all(answer.citations[i].is_cited for i in span.citation_indexes)


def test_fixture_search_count_follows_the_platform():
    assert engines.FixtureEngine("anthropic").answer(EXAMPLE_QUESTION, 1).search_count == 2
    assert engines.FixtureEngine("google").answer(EXAMPLE_QUESTION, 0).search_count is None


def test_placeholder_answer_records_nothing_new():
    answer = engines.FixtureEngine("google").answer("Not a fixture question?", 0)
    assert answer.searches is None and answer.spans is None and answer.usage is None
    assert answer.raw_payload is None


@pytest.mark.parametrize("engine", SUPPORTED_ENGINES)
def test_pre_run_estimate_exists_for_every_default_model(engine):
    # Hand-priced from TYPICAL_USAGE at the list prices in pricing.py.
    expected = {"anthropic": 0.074688, "openai": 0.03633, "google": 0.008651}
    assert engines.per_call_estimate_usd(engine) == pytest.approx(expected[engine], abs=1e-6)
    assert engines.per_call_estimate_usd(engine, "unpriced-model") is None
