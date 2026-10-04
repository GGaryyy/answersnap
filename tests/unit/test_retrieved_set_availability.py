"""The eighth same-direction failure.

`cited/retrieved` is the strongest poisoning signal we have: an attacker can
flood a retrieval index but cannot make a model cite what it flooded, so a
domain that appears in retrieval constantly and in citations rarely is the
pattern. But three of five platforms never expose what they retrieved without
citing — there the ratio is exactly 1.0, and "provably clean" is
indistinguishable from "we cannot see the retrieved set at all".

Hence the flag, per platform, derived per response.
"""

import pytest

from answersnap.providers.anthropic_provider import AnthropicProvider
from answersnap.providers.google_provider import GoogleProvider
from answersnap.providers.openai_provider import OpenAIProvider
from answersnap.providers.perplexity_provider import PerplexityProvider
from answersnap.providers.xai_provider import XAIProvider
from fake_answers import STAMP
from test_provider_parsers import (
    ANTHROPIC_PAYLOAD,
    GOOGLE_PAYLOAD,
    OPENAI_PAYLOAD,
)

PERPLEXITY_PAYLOAD = {
    "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
    "citations": ["https://trlsd.example/a"],
}
XAI_PAYLOAD = {
    "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
    "citations": ["https://trlsd.example/a"],
}


@pytest.mark.parametrize("provider,payload,expected", [
    # These two return the search results alongside what was cited.
    (AnthropicProvider(client=object()), ANTHROPIC_PAYLOAD, True),
    (GoogleProvider(api_key="x"), GOOGLE_PAYLOAD, True),
    # These three return citations only.
    (OpenAIProvider(client=object()), OPENAI_PAYLOAD, False),
    (PerplexityProvider(api_key="x"), PERPLEXITY_PAYLOAD, False),
    (XAIProvider(api_key="x"), XAI_PAYLOAD, False),
])
def test_platforms_declare_whether_the_retrieved_set_is_visible(provider, payload,
                                                                expected):
    answer = provider.parse(payload, STAMP, STAMP)
    assert answer.retrieved_set_available is expected


@pytest.mark.parametrize("provider,payload", [
    (OpenAIProvider(client=object()), OPENAI_PAYLOAD),
    (PerplexityProvider(api_key="x"), PERPLEXITY_PAYLOAD),
    (XAIProvider(api_key="x"), XAI_PAYLOAD),
])
def test_a_platform_that_hides_retrieval_emits_no_retrieved_only_rows(provider, payload):
    answer = provider.parse(payload, STAMP, STAMP)
    # Otherwise the ratio would be computed from a set we never saw.
    assert all(c.is_cited for c in answer.citations)


def test_anthropic_search_without_citations_still_exposes_the_retrieved_set():
    payload = {"content": [{"type": "web_search_tool_result",
                            "content": [{"type": "web_search_result",
                                         "url": "https://spam.example/a"}]},
                           {"type": "text", "text": "我不確定。"}]}
    answer = AnthropicProvider(client=object()).parse(payload, STAMP, STAMP)
    # Retrieved a lot, cited nothing — exactly the shape the poisoning signal
    # looks for, and only visible because the retrieved set is exposed.
    assert answer.retrieved_set_available is True
    assert answer.cited_sources_available is False
    assert [c.is_cited for c in answer.citations] == [False]
