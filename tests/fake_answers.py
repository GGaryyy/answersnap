"""Synthetic provider responses. Schema-identical, entirely invented content.

The Anthropic fakes mirror the real response shape: a source the answer cites
appears in a text block's `citations`, while `web_search_tool_result` carries
every hit the search returned, cited or not.
"""

from datetime import datetime, timezone

from answersnap.providers.base import Provider, ProviderAnswer, RawCitation

# Mixed CJK + emoji on purpose: byte, UTF-16 and code-point offsets all differ
# here, and the whole product's evidence links depend on picking code points.
ANSWER_WITH_EMOJI = "你好🦷 Trellis 是一家診所。Kanbanly 也不錯。"
STAMP = datetime(2026, 8, 22, 3, 0, tzinfo=timezone.utc)
BRAND_CHAR_START = ANSWER_WITH_EMOJI.index("Trellis")


def fake_answer(text=ANSWER_WITH_EMOJI, citations=(), retrieved_only=(),
                platform="anthropic", retrieved_set_available=True):
    stamp = STAMP
    cited = [RawCitation(url=u, title=None, position=i, is_cited=True)
             for i, u in enumerate(citations)]
    retrieved = [RawCitation(url=u, title=None, position=None, is_cited=False)
                 for u in retrieved_only]
    return ProviderAnswer(
        platform=platform,
        model="claude-sonnet-5",
        params={"web_search": True, "temperature": "provider_default"},
        requested_at=stamp,
        observed_at=stamp,
        answer_text=text,
        citations=cited + retrieved,
        retrieved_set_available=retrieved_set_available,
        stop_reason="end_turn",
    )


def anthropic_citation(url, title=None, cited_text=None):
    return {"type": "web_search_result_location", "url": url, "title": title,
            "cited_text": cited_text}


def anthropic_text_block(text, citations=None):
    block = {"type": "text", "text": text}
    if citations is not None:
        block["citations"] = citations
    return block


def anthropic_search_block(results, caller="direct"):
    return {"type": "web_search_tool_result",
            "caller": {"type": caller},
            "content": [{"type": "web_search_result", "url": url, "title": title}
                        for url, title in results]}


class FakeAnthropicResponse:
    """Stands in for the SDK object; parsing happens on its serialised form."""

    def __init__(self, content, stop_reason="end_turn", model="claude-sonnet-5"):
        self._payload = {"content": content, "stop_reason": stop_reason, "model": model}

    def model_dump(self, mode=None):
        return self._payload


class FakeAnthropicClient:
    """Records the kwargs it was called with, so tests can assert on them."""

    def __init__(self, response):
        self._response = response
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class StubProvider(Provider):
    """A provider that answers from fixtures, with its shape counted as verified."""

    platform = "anthropic"

    def __init__(self, answer_factory=None):
        self.asked = []
        self._answer_factory = answer_factory or (lambda: fake_answer())

    @classmethod
    def response_shape_verified(cls):
        return True

    @property
    def model(self):
        return "claude-sonnet-5"

    def parse(self, payload, requested_at, observed_at):
        return self._answer_factory()

    def ask(self, query_text):
        self.asked.append(query_text)
        return self._answer_factory()
