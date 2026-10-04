"""The gate's own logic: what counts as "this parser has seen a real response".

Getting this wrong is worse than having no gate, because a gate that says
"verified" while proving nothing is a gate everyone trusts.
"""

import json

import pytest

from answersnap.providers.anthropic_provider import AnthropicProvider
from answersnap.providers.base import RECORDED_DIR
from test_provider_parsers import ANTHROPIC_PAYLOAD


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """Point the recorded-response directory at a temporary one."""
    monkeypatch.setattr("answersnap.providers.base.RECORDED_DIR", tmp_path)

    def write(payload, platform="anthropic"):
        (tmp_path / f"{platform}.json").write_text(
            json.dumps({"platform": platform, "payload": payload}, ensure_ascii=False),
            encoding="utf-8")

    yield write
    AnthropicProvider.reset_verification_cache()


@pytest.fixture(autouse=True)
def clear_cache():
    AnthropicProvider.reset_verification_cache()
    yield
    AnthropicProvider.reset_verification_cache()


def test_the_recorded_directory_is_absolute():
    # A relative path would make the verdict depend on the working directory,
    # so a job started from elsewhere would silently report every platform
    # unverified.
    assert RECORDED_DIR.is_absolute()


def test_the_recorded_directory_ships_with_the_package():
    # Under tests/ it would be stripped from the container image, and every
    # platform would be unverified in production only.
    assert RECORDED_DIR.parent.name == "providers"
    assert RECORDED_DIR.parent.parent.name == "answersnap"
    assert "tests" not in RECORDED_DIR.parts


def test_a_real_response_marks_the_parser_verified(recorded):
    recorded(ANTHROPIC_PAYLOAD)
    assert AnthropicProvider.response_shape_verified() is True


def test_no_recording_means_unverified(recorded):
    assert AnthropicProvider.response_shape_verified() is False


def test_an_empty_stub_does_not_count_as_verification(recorded):
    # Checking only that a file exists would let this through, and the flag
    # would then mean "someone made a file", not "the parser handled a real
    # response".
    recorded({})
    assert AnthropicProvider.response_shape_verified() is False


def test_a_response_the_parser_finds_no_sources_in_does_not_count(recorded):
    recorded({"content": [{"type": "text", "text": "有答案但沒有來源"}]})
    # A grounded answer with zero citations usually means the parser is reading
    # the wrong field — exactly the failure this gate exists to catch.
    assert AnthropicProvider.response_shape_verified() is False


def test_a_response_with_sources_but_no_text_does_not_count(recorded):
    recorded({"content": [{"type": "web_search_tool_result",
                           "content": [{"type": "web_search_result",
                                        "url": "https://a.tw/x"}]}]})
    assert AnthropicProvider.response_shape_verified() is False


def test_corrupt_json_does_not_count(recorded, tmp_path):
    (tmp_path / "anthropic.json").write_text("{not json", encoding="utf-8")
    assert AnthropicProvider.response_shape_verified() is False


def test_a_payload_that_is_not_an_object_does_not_count(recorded, tmp_path):
    (tmp_path / "anthropic.json").write_text(
        json.dumps({"payload": ["a", "list"]}), encoding="utf-8")
    assert AnthropicProvider.response_shape_verified() is False


def test_the_verdict_is_cached_so_it_is_cheap_to_ask(recorded, tmp_path):
    recorded(ANTHROPIC_PAYLOAD)
    assert AnthropicProvider.response_shape_verified() is True
    (tmp_path / "anthropic.json").unlink()
    # Still true from cache: the gate is consulted on every run, and re-parsing
    # a recording each time would be pure waste.
    assert AnthropicProvider.response_shape_verified() is True
    AnthropicProvider.reset_verification_cache()
    assert AnthropicProvider.response_shape_verified() is False


def test_retrieved_only_sources_do_not_count_as_verification(recorded):
    # The real failure this caught on 2026-08-23: a recording with 14
    # retrieved-only sources and zero citations passed a check that only counted
    # citation rows.
    recorded({"content": [
        {"type": "web_search_tool_result",
         "content": [{"type": "web_search_result", "url": f"https://s{i}.example/a"}
                     for i in range(14)]},
        {"type": "text", "text": "一段有內容的答案", "citations": None},
    ]})
    assert AnthropicProvider.response_shape_verified() is False


def test_one_cited_source_is_enough_to_verify(recorded):
    recorded({"content": [
        {"type": "web_search_tool_result",
         "content": [{"type": "web_search_result", "url": "https://a.tw/x"}]},
        {"type": "text", "text": "答案",
         "citations": [{"type": "web_search_result_location", "url": "https://a.tw/x"}]},
    ]})
    assert AnthropicProvider.response_shape_verified() is True


def test_recordings_ship_redacted():
    # A recording proves the response shape, not its contents. The contents
    # name real businesses that never agreed to appear in a public package,
    # so every shipped recording must have been through the redaction step.
    import re

    for path in sorted(RECORDED_DIR.glob("*.json")):
        recording = json.loads(path.read_text(encoding="utf-8"))
        assert recording.get("redacted"), f"{path.name} is not redacted"
        blob = json.dumps(recording["payload"], ensure_ascii=False)
        hosts = set(re.findall(r"https?://([^/\"]+)", blob))
        unexpected = {h for h in hosts
                      if h != "vertexaisearch.cloud.google.com" and not h.endswith(".example")}
        assert not unexpected, (path.name, unexpected)
        assert not re.search(r"[一-鿿]", blob), f"{path.name} still holds readable CJK text"
