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
    monkeypatch.delenv(engines.KEY_ENV[engine], raising=False)
    status, reason = engines.live_status(engine)
    assert status == engines.SKIPPED_NO_CREDENTIALS
    assert engines.KEY_ENV[engine] in reason


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
