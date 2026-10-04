"""OpenAI Responses API with the web search tool.

The parser here is written from the documented response shape and stays
unverified until a real response is recorded under providers/recorded/. A response
shape that has never been observed must not produce data.
"""

import os
from datetime import datetime, timezone

from answersnap.providers.base import Provider, ProviderAnswer, ProviderError, RawCitation, register

# Consumer-default tier, not the flagship.
DEFAULT_MODEL = "gpt-5.1"
WEB_SEARCH_TOOL = {"type": "web_search"}
MAX_RETRIES = 5


def _output_items(payload):
    items = payload.get("output")
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _text_parts(items):
    parts = []
    for item in items:
        if item.get("type") != "message":
            continue
        for chunk in item.get("content") or []:
            if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                parts.append(chunk.get("text") or "")
    return parts


def _annotations(items):
    for item in items:
        if item.get("type") != "message":
            continue
        for chunk in item.get("content") or []:
            if not isinstance(chunk, dict):
                continue
            for annotation in chunk.get("annotations") or []:
                if isinstance(annotation, dict):
                    yield chunk, annotation


@register
class OpenAIProvider(Provider):
    platform = "openai"

    def __init__(self, model=DEFAULT_MODEL, client=None, search_enabled=True):
        self._model = model
        self._search_enabled = search_enabled
        self._client = client

    @property
    def model(self):
        return self._model

    def _client_or_fail(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ProviderError("openai package not installed") from exc
            self._client = openai.OpenAI(max_retries=MAX_RETRIES)
        return self._client

    def credentials_available(self):
        return bool(self._client is not None or os.environ.get("OPENAI_API_KEY"))

    def _params(self, search_ran=None, model_reported=None):
        # Requested vs observed kept apart: "we asked for search" is not
        # "search happened".
        params = {"web_search": self._cfg("_search_enabled", True),
                  "temperature": "provider_default", "max_retries": MAX_RETRIES,
                  "model_requested": self._cfg("_model", DEFAULT_MODEL)}
        if search_ran is not None:
            params["search_ran"] = search_ran
        if model_reported is not None:
            params["model_reported_by_platform"] = model_reported
        return params

    def request_body(self, query_text):
        return {
            "model": self._model,
            "tools": [WEB_SEARCH_TOOL] if self._search_enabled else [],
            "input": [{"role": "user", "content": query_text}],
        }

    def _cfg(self, name, default):
        # parse() also runs on a bare instance during verification probing.
        return getattr(self, name, default)

    def parse(self, payload, requested_at, observed_at):
        items = _output_items(payload)
        text = "".join(_text_parts(items))

        cited, seen = [], set()
        for _, annotation in _annotations(items):
            if annotation.get("type") != "url_citation":
                continue
            url = annotation.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            cited.append(RawCitation(url=url, title=annotation.get("title"),
                                     position=len(cited), is_cited=True))

        # The Responses API reports citations as annotations on the message.
        # A search call with no annotation channel means we could not observe
        # what was cited — not that nothing was.
        search_ran = any(i.get("type") == "web_search_call" for i in items)
        channel_present = any(isinstance(chunk, dict) and "annotations" in chunk
                              for item in items if item.get("type") == "message"
                              for chunk in item.get("content") or [])
        return ProviderAnswer(
            platform=self.platform,
            model=payload.get("model") or self._cfg("_model", DEFAULT_MODEL),
            params=self._params(search_ran=search_ran,
                                model_reported=bool(payload.get("model"))),
            requested_at=requested_at,
            observed_at=observed_at,
            answer_text=text,
            citations=cited,
            cited_sources_available=bool(self._cfg("_search_enabled", True)
                                         and search_ran and channel_present),
            # The Responses API surfaces annotations (what was cited) but not
            # the full result set behind the search call, so cited/retrieved
            # cannot be computed here at all.
            retrieved_set_available=False,
            # status says completed/incomplete; the reason a response was cut
            # short lives elsewhere, and a truncated answer counts the same as
            # "the brand was not mentioned" if we lose that.
            stop_reason=((payload.get("incomplete_details") or {}).get("reason")
                         or payload.get("status")),
        )

    def ask(self, query_text):
        import openai

        client = self._client_or_fail()
        requested_at = datetime.now(timezone.utc)
        try:
            response = client.responses.create(**self.request_body(query_text))
        except openai.OpenAIError as exc:
            raise ProviderError(f"openai request failed: {type(exc).__name__}") from exc
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        return self.parse(payload, requested_at, datetime.now(timezone.utc))
