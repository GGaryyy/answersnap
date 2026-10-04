"""Parser behaviour per platform, on payload shapes rather than SDK objects.

These use hand-written payloads, which is exactly why they are not enough on
their own: `tests/integration/test_recorded_responses.py` runs the same parsers
against real recorded responses, and the runner refuses any platform that has
none.
"""

import pytest

from answersnap.providers.anthropic_provider import AnthropicProvider
from answersnap.providers.base import ProviderError
from answersnap.providers.google_provider import GoogleProvider
from answersnap.providers.openai_provider import OpenAIProvider
from answersnap.providers.perplexity_provider import PerplexityProvider
from answersnap.providers.xai_provider import XAIProvider
from fake_answers import STAMP

CITED, RETRIEVED = True, False


def _parse(provider, payload):
    return provider.parse(payload, STAMP, STAMP)


def _split(answer):
    return ([c for c in answer.citations if c.is_cited],
            [c for c in answer.citations if not c.is_cited])


# ---------------------------------------------------------------- anthropic
ANTHROPIC_PAYLOAD = {
    "model": "claude-sonnet-5",
    "stop_reason": "end_turn",
    "content": [
        {"type": "web_search_tool_result",
         "content": [{"type": "web_search_result", "url": "https://trlsd.example/a",
                      "title": "範例甲"},
                     {"type": "web_search_result", "url": "https://spam.example/b",
                      "title": "農場"}]},
        {"type": "text", "text": "範例甲不錯。",
         "citations": [{"type": "web_search_result_location",
                        "url": "https://trlsd.example/a", "title": "範例甲",
                        "cited_text": "植牙專科"}]},
    ],
}


def test_anthropic_separates_cited_from_merely_retrieved():
    cited, retrieved = _split(_parse(AnthropicProvider(client=object()), ANTHROPIC_PAYLOAD))
    assert [c.domain for c in cited] == ["trlsd.example"]
    assert [c.domain for c in retrieved] == ["spam.example"]
    assert cited[0].cited_text == "植牙專科"


# ---------------------------------------------------------------- openai
OPENAI_PAYLOAD = {
    "model": "gpt-5.1",
    "status": "completed",
    "output": [
        {"type": "web_search_call", "status": "completed"},
        {"type": "message", "content": [
            {"type": "output_text", "text": "推薦範例甲。",
             "annotations": [{"type": "url_citation", "url": "https://trlsd.example/a",
                              "title": "範例甲"},
                             {"type": "file_citation", "file_id": "f1"}]},
        ]},
    ],
}


def test_openai_reads_url_citations_and_ignores_other_annotations():
    answer = _parse(OpenAIProvider(client=object()), OPENAI_PAYLOAD)
    assert answer.answer_text == "推薦範例甲。"
    assert [c.domain for c in answer.citations] == ["trlsd.example"]
    assert answer.cited_sources_available is True


def test_openai_without_an_annotation_channel_reports_unobservable():
    payload = {"output": [{"type": "web_search_call"},
                          {"type": "message",
                           "content": [{"type": "output_text", "text": "答案"}]}]}
    assert _parse(OpenAIProvider(client=object()), payload).cited_sources_available is False


# ---------------------------------------------------------------- perplexity
def test_perplexity_merges_both_source_keys_without_duplicates():
    payload = {
        "model": "sonar",
        "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
        "search_results": [{"url": "https://trlsd.example/a", "title": "範例甲"}],
        "citations": ["https://trlsd.example/a", "https://sample-clinic.example/c"],
    }
    answer = _parse(PerplexityProvider(api_key="x"), payload)
    assert [c.domain for c in answer.citations] == ["trlsd.example", "sample-clinic.example"]
    assert [c.position for c in answer.citations] == [0, 1]
    assert answer.cited_sources_available is True


def test_perplexity_without_a_source_key_reports_unobservable():
    payload = {"choices": [{"message": {"content": "答案"}}]}
    assert _parse(PerplexityProvider(api_key="x"), payload).cited_sources_available is False


# ---------------------------------------------------------------- google
GOOGLE_PAYLOAD = {
    "modelVersion": "gemini-3-flash",
    "candidates": [{
        "finishReason": "STOP",
        "content": {"parts": [{"text": "推薦範例甲。"}]},
        "groundingMetadata": {
            "groundingChunks": [
                {"web": {"uri": "https://trlsd.example/a", "title": "範例甲"}},
                {"web": {"uri": "https://spam.example/b", "title": "農場"}},
            ],
            "groundingSupports": [{"groundingChunkIndices": [0]}],
        },
    }],
}


def test_google_uses_grounding_supports_to_tell_cited_from_retrieved():
    cited, retrieved = _split(_parse(GoogleProvider(api_key="x"), GOOGLE_PAYLOAD))
    assert [c.domain for c in cited] == ["trlsd.example"]
    assert [c.domain for c in retrieved] == ["spam.example"]


def test_google_without_grounding_supports_reports_unobservable():
    payload = {"candidates": [{"content": {"parts": [{"text": "答案"}]},
                               "groundingMetadata": {"groundingChunks": []}}]}
    assert _parse(GoogleProvider(api_key="x"), payload).cited_sources_available is False


# ---------------------------------------------------------------- xai
def test_xai_marks_x_posts_as_social_sources():
    payload = {
        "model": "grok-4",
        "choices": [{"message": {"content": "有人在討論。"}, "finish_reason": "stop"}],
        "citations": ["https://x.com/someone/status/1", "https://sample-clinic.example/c"],
    }
    answer = _parse(XAIProvider(api_key="x"), payload)
    # Social sources need auth to fetch and vanish far more often than pages;
    # treating them as ordinary web links would record deletions as fetch errors.
    assert [c.source_type for c in answer.citations] == ["social_post", "web"]


def test_xai_without_citations_key_reports_unobservable():
    payload = {"choices": [{"message": {"content": "答案"}}]}
    assert _parse(XAIProvider(api_key="x"), payload).cited_sources_available is False


# ---------------------------------------------------------------- shared
@pytest.mark.parametrize("provider", [
    AnthropicProvider(client=object()),
    OpenAIProvider(client=object()),
    PerplexityProvider(api_key="x"),
    GoogleProvider(api_key="x"),
    XAIProvider(api_key="x"),
])
def test_every_parser_survives_an_empty_payload(provider):
    answer = _parse(provider, {})
    assert answer.answer_text == ""
    assert answer.citations == []
    # Nothing observed means nothing claimed.
    assert answer.cited_sources_available is False


@pytest.mark.parametrize("provider", [
    AnthropicProvider(client=object()),
    OpenAIProvider(client=object()),
    PerplexityProvider(api_key="x"),
    GoogleProvider(api_key="x"),
    XAIProvider(api_key="x"),
])
def test_no_parser_sets_a_temperature(provider):
    # Consumers never get temperature 0; pinning it would measure another world.
    body = provider.request_body("問法")
    assert "temperature" not in body
    assert "system" not in body


class FakeOpenAIResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode=None):
        return self._payload


class FakeOpenAIClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOpenAIResponse(self._payload)


def test_openai_live_call_goes_through_the_same_parser():
    client = FakeOpenAIClient(OPENAI_PAYLOAD)
    answer = OpenAIProvider(client=client).ask("問法")

    assert [c.domain for c in answer.citations] == ["trlsd.example"]
    sent = client.calls[0]
    assert sent["input"] == [{"role": "user", "content": "問法"}]
    assert "temperature" not in sent
    assert "system" not in sent


def test_openai_errors_become_provider_errors():
    import openai

    class BrokenClient:
        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            raise openai.APIConnectionError(request=None)

    with pytest.raises(ProviderError, match="openai request failed"):
        OpenAIProvider(client=BrokenClient()).ask("問法")


def test_search_can_be_switched_off_for_every_platform():
    # Without search there is no citation channel at all, which must read as
    # "unobservable" rather than "nothing cited".
    for provider in (AnthropicProvider(client=object(), search_enabled=False),
                     OpenAIProvider(client=object(), search_enabled=False),
                     GoogleProvider(api_key="x", search_enabled=False),
                     XAIProvider(api_key="x", search_enabled=False)):
        body = provider.request_body("問法")
        assert not body.get("tools")
        assert "search_parameters" not in body


def test_openai_reports_the_real_truncation_reason():
    payload = {"status": "incomplete",
               "incomplete_details": {"reason": "max_output_tokens"},
               "output": [{"type": "message",
                           "content": [{"type": "output_text", "text": "被截斷的"}]}]}
    # A truncated answer counts the same as "the brand was not mentioned", so
    # losing the reason loses the ability to tell those apart.
    assert _parse(OpenAIProvider(client=object()), payload).stop_reason == "max_output_tokens"


def test_xai_asks_for_citations_explicitly():
    body = XAIProvider(api_key="k").request_body("問法")
    # If return_citations defaults off, Grok's citation data disappears entirely
    # — on the platform where smearing surfaces first.
    assert body["search_parameters"]["return_citations"] is True
    assert {s["type"] for s in body["search_parameters"]["sources"]} == {"web", "x"}


def test_perplexity_records_an_empty_citation_channel():
    payload = {"choices": [{"message": {"content": "答案"}}], "citations": []}
    answer = _parse(PerplexityProvider(api_key="x"), payload)
    assert answer.cited_sources_available is True
    assert answer.params["empty_citation_channel"] is True


@pytest.mark.parametrize("provider,payload", [
    (AnthropicProvider(client=object()), ANTHROPIC_PAYLOAD),
    (OpenAIProvider(client=object()), OPENAI_PAYLOAD),
    (GoogleProvider(api_key="x"), GOOGLE_PAYLOAD),
])
def test_what_we_asked_for_is_stored_apart_from_what_happened(provider, payload):
    answer = _parse(provider, payload)
    # "We asked for search" is not "search happened", and the model we requested
    # is not necessarily the model that answered. Collapsing either pair turns a
    # configuration value into a recorded measurement.
    assert answer.params["model_requested"]
    assert answer.params["model_reported_by_platform"] is True
    assert answer.params["search_ran"] is True


@pytest.mark.parametrize("provider", [
    AnthropicProvider(client=object()),
    OpenAIProvider(client=object()),
    GoogleProvider(api_key="x"),
])
def test_a_platform_that_reports_no_model_is_not_credited_with_ours(provider):
    answer = _parse(provider, {})
    # The model field falls back to our request so the row can be stored, but
    # the row says so rather than passing it off as the platform's answer.
    assert answer.params["model_reported_by_platform"] is False
    assert answer.params["model_requested"] == answer.model
    assert answer.params["search_ran"] is False


def test_google_sees_through_its_own_redirect_links():
    # Real grounding responses never carry the site in the uri: every link is
    # a vertexaisearch redirect, and the domain is only in the title. Read
    # naively, no clinic's site ever matches and "cited 0 times" is invented.
    redirect = ("https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
                "AUZIYQHNSYjOX5ttuhfh01W")
    payload = {
        "modelVersion": "gemini-3.7-flash",
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": "推薦蒔美。"}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": redirect, "title": "smile-dental.tw"}},
                    {"web": {"uri": redirect + "x", "title": "植牙攻略 2026"}},
                ],
                "groundingSupports": [{"groundingChunkIndices": [0, 1]}],
            },
        }],
    }
    cited, _ = _split(_parse(GoogleProvider(api_key="x"), payload))
    assert cited[0].domain == "smile-dental.tw"
    assert cited[0].url == redirect  # the click-through link is kept as-is
    # A page title is not a site; the redirector stays visible rather than
    # being turned into a guessed domain.
    assert cited[1].domain == "vertexaisearch.cloud.google.com"
