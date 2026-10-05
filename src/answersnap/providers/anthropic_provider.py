import os
from datetime import datetime, timezone

import anthropic

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

# Consumer-default tier, not the flagship: we measure the world most people see.
# This is a measurement decision, not a cost decision — if the consumer default
# ever becomes the flagship, this changes even though it costs more.
DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
# allowed_callers must be "direct". This tool version defaults to running search
# inside code execution (dynamic filtering), which filters results before they
# reach the model — and in that mode the answer carries NO citations at all.
# Verified against a real response on 2026-08-23: every text block came back with
# citations=None. Citations are the product, so we take the direct path.
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search",
                   "allowed_callers": ["direct"]}
# The SDK backs off on 429 and 5xx. A rate limit that ends the call would turn a
# recoverable delay into a permanent hole in the time series.
MAX_RETRIES = 5


def _cited_sources(blocks):
    """URLs the answer actually cites, in the order they appear in the answer."""
    cited, seen = [], set()
    for block in blocks:
        if block.get("type") != "text":
            continue
        for citation in block.get("citations") or []:
            url = citation.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            cited.append(RawCitation(url=url, title=citation.get("title"),
                                     position=len(cited), is_cited=True,
                                     cited_text=citation.get("cited_text")))
    return cited, {c.url: i for i, c in enumerate(cited)}


def _searches(blocks):
    """Queries the model sent, each with the result count of its result block."""
    results = {b.get("tool_use_id"): b.get("content") for b in blocks
               if b.get("type") == "web_search_tool_result"}
    searches = []
    for block in blocks:
        if block.get("type") != "server_tool_use" or block.get("name") != "web_search":
            continue
        query = (block.get("input") or {}).get("query")
        if not isinstance(query, str):
            continue
        found = results.get(block.get("id"))
        searches.append(SearchQuery(
            query=query, results_count=len(found) if isinstance(found, list) else None))
    return searches


def _spans(blocks, index_by_url, cited_flags):
    """Each cited text block is one span; its offset is the text before it.

    The API gives no offsets, but the answer text is exactly the text blocks
    joined in order, so the cumulative length places every block.
    """
    if not any(b.get("type") == "text" and isinstance(b.get("citations"), list)
               for b in blocks):
        return None, 0
    candidates, offset = [], 0
    for block in blocks:
        if block.get("type") != "text":
            continue
        length = len(block.get("text") or "")
        citations = block.get("citations")
        if isinstance(citations, list) and citations:
            indexes = [index_by_url.get((c or {}).get("url")) for c in citations
                       if isinstance(c, dict)]
            candidates.append((offset, offset + length, indexes))
        offset += length
    return valid_spans(candidates, offset, cited_flags)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _usage(payload):
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    details = _dict(usage.get("output_tokens_details"))
    server = _dict(usage.get("server_tool_use"))
    search_count = server.get("web_search_requests")
    # Thinking tokens are billed as output and already inside output_tokens.
    summary = usage_summary(usage.get("input_tokens"), usage.get("output_tokens"),
                            details.get("thinking_tokens"), None, True, usage)
    return summary, search_count if isinstance(search_count, int) else None


def _retrieved_only_sources(blocks, cited_urls):
    """Search hits the answer did not cite — kept, but never counted as citations."""
    retrieved, seen = [], set()
    for block in blocks:
        if block.get("type") != "web_search_tool_result":
            continue
        results = block.get("content")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get("url")
            if not url or url in cited_urls or url in seen:
                continue
            seen.add(url)
            retrieved.append(RawCitation(url=url, title=result.get("title"),
                                         position=None, is_cited=False))
    return retrieved


def _caller_types(blocks):
    """Which caller ran the searches in this response.

    Not a constant: it comes from a tool setting whose default is dynamic
    filtering, and under that default the answer carries no citations at all.
    Whether citations are observable is therefore a property of THIS call, and
    the evidence for it has to live in the row — a downstream check cannot read
    our configuration file.
    """
    types = set()
    for block in blocks:
        if block.get("type") != "web_search_tool_result":
            continue
        caller = block.get("caller")
        types.add((caller or {}).get("type") or "unknown")
    return sorted(types)


def _citations_were_observable(blocks, search_enabled):
    """Whether THIS call could have told us which sources were cited.

    A class-level capability flag would be wrong per request: without search
    there are no citations at all, and some parameter combinations do not carry
    the citation channel. Getting this wrong recreates the seventh
    same-direction failure — a blind spot that looks like a poisoning signal.
    """
    if not search_enabled:
        return False
    search_ran = any(b.get("type") == "web_search_tool_result" for b in blocks)
    # The key is always present; its VALUE is what says whether the channel
    # exists. Under dynamic filtering every block carries citations=None, and
    # treating key-presence as availability would report a total blind spot as
    # "this platform cited nothing" — the seventh same-direction failure, from
    # the inside.
    channel_present = any(b.get("type") == "text" and isinstance(b.get("citations"), list)
                          for b in blocks)
    return search_ran and channel_present


@register
class AnthropicProvider(Provider):
    platform = "anthropic"

    def __init__(self, model=DEFAULT_MODEL, client=None, search_enabled=True):
        self._model = model
        self._search_enabled = search_enabled
        # Built on first use: constructing eagerly makes a missing key an
        # exception at import-ish time rather than a reported skip.
        self._client = client

    def _client_or_fail(self):
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        return self._client

    @property
    def model(self):
        return self._model

    def credentials_available(self):
        return bool(self._client is not None or os.environ.get("ANTHROPIC_API_KEY"))

    def _params(self, callers=None, search_ran=None, model_reported=None):
        # Requested settings and observed outcomes are kept apart throughout.
        # Recording "we asked for search" as if it were "search happened" is the
        # same error as recording a config value as a measurement.
        params = {"web_search": self._cfg("_search_enabled", True),
                  "max_tokens": MAX_TOKENS, "temperature": "provider_default",
                  "max_retries": MAX_RETRIES,
                  "search_tool": WEB_SEARCH_TOOL["type"],
                  "allowed_callers": WEB_SEARCH_TOOL["allowed_callers"],
                  "model_requested": self._cfg("_model", DEFAULT_MODEL)}
        if callers is not None:
            params["search_callers"] = callers
        if search_ran is not None:
            params["search_ran"] = search_ran
        if model_reported is not None:
            # False means the model field is our request echoed back, not the
            # platform telling us what answered.
            params["model_reported_by_platform"] = model_reported
        return params

    def request_body(self, query_text):
        return {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "tools": [WEB_SEARCH_TOOL] if self._search_enabled else [],
            "messages": [{"role": "user", "content": query_text}],
        }

    def _cfg(self, name, default):
        # parse() also runs on a bare instance during verification probing.
        return getattr(self, name, default)

    def parse(self, payload, requested_at, observed_at):
        blocks = [b for b in payload.get("content", []) if isinstance(b, dict)]
        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
        cited, index_by_url = _cited_sources(blocks)
        citations = cited + _retrieved_only_sources(blocks, set(index_by_url))
        spans, dropped = _spans(blocks, index_by_url, [c.is_cited for c in citations])
        usage, search_count = _usage(payload)
        search_ran = any(b.get("type") == "web_search_tool_result" for b in blocks)
        return ProviderAnswer(
            platform=self.platform,
            model=payload.get("model") or self._cfg("_model", DEFAULT_MODEL),
            params=self._params(
                callers=_caller_types(blocks),
                search_ran=search_ran,
                model_reported=bool(payload.get("model"))),
            requested_at=requested_at,
            observed_at=observed_at,
            answer_text=text,
            citations=citations,
            cited_sources_available=_citations_were_observable(
                blocks, self._cfg("_search_enabled", True)),
            # The search tool result block lists everything retrieved, cited or
            # not, so the retrieved set is visible whenever search ran.
            retrieved_set_available=search_ran,
            stop_reason=payload.get("stop_reason"),
            searches=_searches(blocks),
            search_count=search_count,
            spans=spans,
            spans_dropped=dropped,
            usage=usage,
            raw_payload=payload,
        )

    def ask(self, query_text):
        requested_at = datetime.now(timezone.utc)
        try:
            response = self._client_or_fail().messages.create(
                **self.request_body(query_text))
        except anthropic.APIError as exc:
            raise ProviderError(f"anthropic request failed: {type(exc).__name__}") from exc
        # Parse the serialised payload, so production runs through exactly the
        # code path the recorded-response contract test exercises.
        return self.parse(to_payload(response), requested_at, datetime.now(timezone.utc))
