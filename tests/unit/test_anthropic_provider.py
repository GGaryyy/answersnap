import anthropic
import pytest

from answersnap.providers import get_provider, registered_platforms
from answersnap.providers.anthropic_provider import AnthropicProvider
from answersnap.providers.base import ProviderError
from fake_answers import (
    FakeAnthropicClient,
    FakeAnthropicResponse,
    anthropic_citation,
    anthropic_search_block,
    anthropic_text_block,
)

QUERY = "雙北植牙推薦的醫生或診所？"


def _client(blocks):
    return FakeAnthropicClient(FakeAnthropicResponse(blocks))


def test_measurement_request_carries_no_system_prompt_and_no_temperature():
    client = _client([anthropic_text_block("答案")])
    AnthropicProvider(client=client).ask(QUERY)
    sent = client.calls[0]
    # A system prompt or a pinned temperature would measure a world no consumer
    # ever sees; the query must go out exactly as a person would type it.
    assert "system" not in sent
    assert "temperature" not in sent
    assert sent["messages"] == [{"role": "user", "content": QUERY}]


def test_each_call_is_a_fresh_conversation():
    client = _client([anthropic_text_block("答案")])
    provider = AnthropicProvider(client=client)
    provider.ask(QUERY)
    provider.ask(QUERY)
    assert all(len(call["messages"]) == 1 for call in client.calls)


def test_text_blocks_are_concatenated_untouched():
    client = _client([anthropic_text_block("  第一段"), anthropic_text_block("第二段  ")])
    answer = AnthropicProvider(client=client).ask(QUERY)
    assert answer.answer_text == "  第一段第二段  "


def test_only_sources_the_answer_cites_count_as_citations():
    client = _client([
        anthropic_search_block([("https://trlsd.example/a", None), ("https://spam-directory.example/b", None), ("https://sample-clinic.example/c", None)]),
        anthropic_text_block("範例甲很不錯。", citations=[
            anthropic_citation("https://trlsd.example/a", "範例甲", cited_text="植牙專科")]),
        anthropic_text_block("樣本也是。", citations=[anthropic_citation("https://sample-clinic.example/c", "樣本")]),
    ])
    answer = AnthropicProvider(client=client).ask(QUERY)

    cited = [c for c in answer.citations if c.is_cited]
    retrieved = [c for c in answer.citations if not c.is_cited]
    # A merely retrieved domain counted as a citation would inflate
    # citation_rate and the source panel — the numbers we sell.
    assert [(c.domain, c.position) for c in cited] == [("trlsd.example", 0), ("sample-clinic.example", 1)]
    assert [c.domain for c in retrieved] == ["spam-directory.example"]
    assert all(c.position is None for c in retrieved)
    assert cited[0].cited_text == "植牙專科"


def test_position_follows_answer_order_not_search_order():
    client = _client([
        anthropic_search_block([("https://first-in-search.example/x", None), ("https://second-in-search.example/y", None)]),
        anthropic_text_block("先講第二個。", citations=[anthropic_citation("https://second-in-search.example/y")]),
        anthropic_text_block("再講第一個。", citations=[anthropic_citation("https://first-in-search.example/x")]),
    ])
    cited = [c for c in AnthropicProvider(client=client).ask(QUERY).citations if c.is_cited]
    assert [c.domain for c in cited] == ["second-in-search.example", "first-in-search.example"]


def test_a_source_cited_twice_is_recorded_once():
    url = "https://trlsd.example/a"
    client = _client([anthropic_text_block("甲", citations=[anthropic_citation(url)]),
                      anthropic_text_block("乙", citations=[anthropic_citation(url)])])
    answer = AnthropicProvider(client=client).ask(QUERY)
    assert len(answer.citations) == 1


def test_text_block_without_citations_is_handled():
    client = _client([anthropic_text_block("沒有引用的答案", citations=None)])
    assert AnthropicProvider(client=client).ask(QUERY).citations == []


def test_search_result_without_url_is_skipped():
    client = _client([anthropic_search_block([(None, "無網址")])])
    assert AnthropicProvider(client=client).ask(QUERY).citations == []


def test_unknown_platform_is_refused_rather_than_guessed():
    with pytest.raises(ProviderError):
        get_provider("bing")


def test_anthropic_adapter_is_registered():
    assert "anthropic" in registered_platforms()


class FailingClient:
    def __init__(self, exc):
        self._exc = exc
        self.messages = self

    def create(self, **kwargs):
        raise self._exc


class _ValidationError(anthropic.APIError):
    """Stands in for APIResponseValidationError, which needs a live response."""

    def __init__(self):
        super().__init__("invalid schema", request=None, body=None)


@pytest.mark.parametrize("exc_factory", [
    lambda: anthropic.APIConnectionError(request=None),
    _ValidationError,
])
def test_any_sdk_error_becomes_a_provider_error(exc_factory):
    # Not every SDK error is an APIStatusError; letting one escape would abort
    # the whole run instead of costing a single sample.
    with pytest.raises(ProviderError, match="anthropic request failed"):
        AnthropicProvider(client=FailingClient(exc_factory())).ask(QUERY)


def test_default_client_is_configured_to_retry_rate_limits(monkeypatch):
    # A 429 that ends the call turns a recoverable delay into a permanent hole
    # in the time series, so the default client must back off and retry.
    captured = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return FakeAnthropicClient(FakeAnthropicResponse([]))

    monkeypatch.setattr(anthropic, "Anthropic", fake_anthropic)
    # The client is built lazily, so a missing key is a reported skip rather
    # than an exception at construction.
    AnthropicProvider()._client_or_fail()
    assert captured["max_retries"] >= 3


def test_search_result_block_with_unexpected_content_is_ignored():
    class OddBlock:
        type = "web_search_tool_result"
        content = "not a list"

    assert AnthropicProvider(client=_client([OddBlock()])).ask(QUERY).citations == []


def test_citation_availability_is_measured_per_call_not_declared():
    # Search ran and the citation channel is present: we could have seen them.
    client = _client([anthropic_search_block([("https://a.tw/x", None)]),
                      anthropic_text_block("答案", citations=[anthropic_citation("https://a.tw/x")])])
    assert AnthropicProvider(client=client).ask(QUERY).cited_sources_available is True


def test_a_call_without_search_reports_citations_as_unobservable():
    # No search means no citation channel at all; claiming availability here
    # would make the blind spot look like "nothing was cited".
    client = _client([anthropic_text_block("答案")])
    answer = AnthropicProvider(client=client, search_enabled=False).ask(QUERY)
    assert answer.cited_sources_available is False


def test_a_response_without_the_citation_channel_is_marked_unobservable():
    client = _client([anthropic_search_block([("https://a.tw/x", None)]),
                      {"type": "text", "text": "答案"}])
    answer = AnthropicProvider(client=client).ask(QUERY)
    assert answer.cited_sources_available is False
    # The retrieved source is still stored; only the ratio must not use it.
    assert [c.domain for c in answer.citations] == ["a.tw"]


def test_search_that_returned_nothing_is_not_read_as_citations_available():
    client = _client([anthropic_text_block("我不知道", citations=None)])
    assert AnthropicProvider(client=client).ask(QUERY).cited_sources_available is False


def test_search_runs_directly_so_the_answer_carries_citations():
    # web_search_20260209 defaults to running inside code execution (dynamic
    # filtering), and in that mode the answer comes back with no citations at
    # all — verified against a real response, 2026-08-23. Citations are the
    # product, so the direct path is not optional.
    body = AnthropicProvider(client=object()).request_body("問法")
    assert body["tools"][0]["allowed_callers"] == ["direct"]


def test_a_null_citations_field_is_a_blind_spot_not_an_absence_of_citations():
    # Every text block carries the key; only a list value means the channel is
    # really there. Reading key-presence as availability would report a total
    # blind spot as "this platform cited nothing".
    client = _client([anthropic_search_block([("https://a.tw/x", None)]),
                      {"type": "text", "text": "答案", "citations": None}])
    answer = AnthropicProvider(client=client).ask(QUERY)
    assert answer.cited_sources_available is False
    assert [c.is_cited for c in answer.citations] == [False]


def test_an_empty_citations_list_means_the_channel_worked_and_nothing_was_cited():
    client = _client([anthropic_search_block([("https://a.tw/x", None)]),
                      {"type": "text", "text": "答案", "citations": []}])
    answer = AnthropicProvider(client=client).ask(QUERY)
    assert answer.cited_sources_available is True


def test_the_caller_that_actually_ran_is_recorded_on_every_row():
    client = _client([anthropic_search_block([("https://a.tw/x", None)]),
                      anthropic_text_block("答案", citations=[anthropic_citation("https://a.tw/x")])])
    answer = AnthropicProvider(client=client).ask(QUERY)
    # The tool setting says what we asked for; this says what happened. A
    # downstream check reads rows, not our configuration file.
    assert answer.params["allowed_callers"] == ["direct"]
    assert answer.params["search_callers"] == ["direct"]


def test_a_search_that_ran_under_dynamic_filtering_is_visible_in_the_row():
    blocks = [anthropic_search_block([("https://a.tw/x", None)]),
              anthropic_text_block("答案", citations=None)]
    blocks[0]["caller"] = {"type": "code_execution_20260120"}
    answer = AnthropicProvider(client=_client(blocks)).ask(QUERY)
    # This is the mode that silently removes citations. If it ever comes back —
    # an SDK upgrade, someone trimming tokens — the rows say so.
    assert answer.params["search_callers"] == ["code_execution_20260120"]
    assert answer.cited_sources_available is False
