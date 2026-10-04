"""Every platform's failure must cost one sample, never the whole run."""

import pytest

from answersnap.providers.base import ProviderError
from answersnap.providers.google_provider import GoogleProvider
from answersnap.providers.http import HttpError
from answersnap.providers.perplexity_provider import PerplexityProvider
from answersnap.providers.xai_provider import XAIProvider

QUERY = "雙北植牙推薦的醫生或診所？"


def _failing_transport(url, body, headers=None):
    raise HttpError("HTTP 429")


@pytest.mark.parametrize("provider_factory,name", [
    (lambda: PerplexityProvider(api_key="k", transport=_failing_transport), "perplexity"),
    (lambda: GoogleProvider(api_key="k", transport=_failing_transport), "google"),
    (lambda: XAIProvider(api_key="k", transport=_failing_transport), "xai"),
])
def test_transport_failures_become_provider_errors(provider_factory, name):
    with pytest.raises(ProviderError, match=f"{name} request failed"):
        provider_factory().ask(QUERY)


@pytest.mark.parametrize("provider_factory,env", [
    (lambda: PerplexityProvider(api_key=None), "PERPLEXITY_API_KEY"),
    (lambda: GoogleProvider(api_key=None), "GOOGLE_API_KEY"),
    (lambda: XAIProvider(api_key=None), "XAI_API_KEY"),
])
def test_a_missing_key_says_which_key(provider_factory, env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ProviderError, match=env):
        provider_factory().ask(QUERY)


def test_a_successful_call_goes_through_the_shared_parser():
    def transport(url, body, headers=None):
        return {"model": "sonar",
                "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
                "citations": ["https://trlsd.example/a"]}

    answer = PerplexityProvider(api_key="k", transport=transport).ask(QUERY)
    assert answer.answer_text == "答案"
    assert [c.domain for c in answer.citations] == ["trlsd.example"]
    assert answer.observed_at >= answer.requested_at


def test_google_search_grounding_is_requested_by_default():
    body = GoogleProvider(api_key="k").request_body(QUERY)
    assert body["tools"] == [{"google_search": {}}]
    assert body["contents"][0]["parts"][0]["text"] == QUERY


def test_xai_asks_for_live_search():
    # Grok's value here is live X opinion; without search it measures nothing new.
    body = XAIProvider(api_key="k").request_body(QUERY)
    assert body["search_parameters"]["mode"] == "on"
