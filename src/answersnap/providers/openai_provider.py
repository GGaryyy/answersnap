"""OpenAI Responses API with the web search tool.

The parser here is written from the documented response shape and stays
unverified until a real response is recorded under providers/recorded/. A response
shape that has never been observed must not produce data.
"""

import os
from datetime import datetime, timezone

from answersnap.providers.base import (
    Provider,
    ProviderAnswer,
    ProviderError,
    RawCitation,
    SearchQuery,
    register,
    to_payload,
)
from answersnap.providers.spans import usage_summary, valid_spans

# Consumer-default tier, not the flagship.
DEFAULT_MODEL = "gpt-5.1"
WEB_SEARCH_TOOL = {"type": "web_search"}
MAX_RETRIES = 5


def _output_items(payload):
    items = payload.get("output")
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _chunks(items):
    """(base offset, chunk) for every output_text chunk, in answer order.

    Annotation offsets count from the start of their own chunk, so text and
    annotations are read from this one iteration: two separate walks could
    disagree on which chunks count, and every offset would drift.
    """
    base = 0
    for item in items:
        if item.get("type") != "message":
            continue
        for chunk in item.get("content") or []:
            if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                yield base, chunk
                base += len(chunk.get("text") or "")


def _cited_and_spans(chunks):
    """(cited sources, {(start, end): [citation index]}, annotations not placeable)."""
    cited, index_by_url, ranges, unplaced = [], {}, {}, 0
    for base, chunk in chunks:
        for annotation in chunk.get("annotations") or []:
            if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                continue
            url = annotation.get("url")
            if not url:
                continue
            if url not in index_by_url:
                index_by_url[url] = len(cited)
                cited.append(RawCitation(url=url, title=annotation.get("title"),
                                         position=len(cited), is_cited=True))
            start, end = annotation.get("start_index"), annotation.get("end_index")
            if not (_is_offset(start) and _is_offset(end)):
                unplaced += 1
                continue
            # Several sources backing the same range become one span.
            ranges.setdefault((base + start, base + end), []).append(index_by_url[url])
    return cited, ranges, unplaced


def _is_offset(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _searches(items):
    searches = []
    for item in items:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        if action.get("type") not in (None, "search"):
            continue
        queries = action.get("queries")
        if not isinstance(queries, list):
            queries = [action.get("query")]
        searches.extend(SearchQuery(query=q) for q in queries if isinstance(q, str))
    return searches


def _dict(value):
    return value if isinstance(value, dict) else {}


def _usage(payload):
    usage = payload.get("usage")
    tool_usage = _dict(_dict(payload.get("tool_usage")).get("web_search"))
    search_count = tool_usage.get("num_requests")
    search_count = search_count if isinstance(search_count, int) else None
    if not isinstance(usage, dict):
        return None, search_count
    details = _dict(usage.get("output_tokens_details"))
    # reasoning_tokens are billed as output and already inside output_tokens.
    summary = usage_summary(usage.get("input_tokens"), usage.get("output_tokens"),
                            details.get("reasoning_tokens"), usage.get("total_tokens"),
                            True, usage)
    return summary, search_count


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
        chunks = list(_chunks(items))
        text = "".join(chunk.get("text") or "" for _, chunk in chunks)
        cited, ranges, unplaced = _cited_and_spans(chunks)
        usage, search_count = _usage(payload)

        # The Responses API reports citations as annotations on the message.
        # A search call with no annotation channel means we could not observe
        # what was cited — not that nothing was.
        search_ran = any(i.get("type") == "web_search_call" for i in items)
        channel_present = any(isinstance(chunk, dict) and "annotations" in chunk
                              for item in items if item.get("type") == "message"
                              for chunk in item.get("content") or [])
        spans, dropped = (valid_spans([(s, e, i) for (s, e), i in ranges.items()],
                                      len(text), [True] * len(cited))
                          if channel_present else (None, 0))
        dropped += unplaced if channel_present else 0
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
            searches=_searches(items),
            search_count=search_count,
            spans=spans,
            spans_dropped=dropped,
            usage=usage,
            raw_payload=payload,
        )

    def ask(self, query_text):
        import openai

        client = self._client_or_fail()
        requested_at = datetime.now(timezone.utc)
        try:
            response = client.responses.create(**self.request_body(query_text))
        except openai.OpenAIError as exc:
            raise ProviderError(f"openai request failed: {type(exc).__name__}") from exc
        return self.parse(to_payload(response), requested_at, datetime.now(timezone.utc))
