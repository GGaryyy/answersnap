"""Gemini with Google Search grounding.

Grounding metadata separates the chunks retrieved from the chunks the answer
actually leaned on, so both survive here. Parser unverified until recorded.
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
)
from answersnap.providers.citation_host import resolve_domain
from answersnap.providers.http import HttpError, post_json
from answersnap.providers.spans import code_point_at, usage_summary, utf8_offsets, valid_spans

# Consumer-default tier: the model behind the free Gemini app. Pinned to an
# exact id, never a floating alias like gemini-flash-latest — an alias would
# swap the model underneath us mid-series, and the appearance-rate history
# either breaks silently or reports a change we cannot date.
DEFAULT_MODEL = "gemini-3.7-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# GEMINI_API_KEY is the name Google documents for the Gemini API; the older
# GOOGLE_API_KEY still works, so a key set under either name is found.
API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
API_KEY_ENV = API_KEY_ENVS[0]


def _first_candidate(payload):
    candidates = payload.get("candidates") or []
    return candidates[0] if candidates and isinstance(candidates[0], dict) else {}


def _grounding_chunks(candidate):
    metadata = candidate.get("groundingMetadata") or {}
    chunks = metadata.get("groundingChunks") or []
    return metadata, [c for c in chunks if isinstance(c, dict)]


def _web(chunk):
    web = chunk.get("web")
    return web if isinstance(web, dict) else {}


def _supports(metadata):
    return [s for s in metadata.get("groundingSupports") or [] if isinstance(s, dict)]


def _support_candidate(support, text, offsets, index_by_chunk):
    """(start, end, indexes) in code points, or None if it cannot be placed.

    Segment offsets are UTF-8 bytes. The segment also repeats its own text, so
    a span whose slice does not reproduce it is dropped: a wrong guess at the
    unit then shows up as missing spans, never as spans in the wrong place.
    """
    segment = support.get("segment")
    if not isinstance(segment, dict):
        return None
    start = code_point_at(offsets, segment.get("startIndex", 0))
    end = code_point_at(offsets, segment.get("endIndex"))
    if start is None or end is None:
        return None
    quoted = segment.get("text")
    if isinstance(quoted, str) and text[start:end].strip() != quoted.strip():
        return None
    chunk_indexes = support.get("groundingChunkIndices")
    if not isinstance(chunk_indexes, list):
        return None
    indexes = [index_by_chunk.get(i) if isinstance(i, int) else None for i in chunk_indexes]
    return start, end, indexes


def _spans(metadata, text, index_by_chunk, cited_flags):
    if "groundingSupports" not in metadata:
        return None, 0
    offsets = utf8_offsets(text)
    candidates, unplaced = [], 0
    for support in _supports(metadata):
        candidate = _support_candidate(support, text, offsets, index_by_chunk)
        if candidate is None:
            unplaced += 1
        else:
            candidates.append(candidate)
    spans, dropped = valid_spans(candidates, len(text), cited_flags)
    return spans, dropped + unplaced


def _searches(metadata):
    queries = metadata.get("webSearchQueries")
    if not isinstance(queries, list):
        return None
    return [SearchQuery(query=q) for q in queries if isinstance(q, str)]


def _usage(payload):
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    # candidatesTokenCount excludes thinking; totalTokenCount includes it.
    return usage_summary(usage.get("promptTokenCount"), usage.get("candidatesTokenCount"),
                         usage.get("thoughtsTokenCount"), usage.get("totalTokenCount"),
                         False, usage)


def _key_from_environment():
    return next((os.environ[name] for name in API_KEY_ENVS if os.environ.get(name)), None)


@register
class GoogleProvider(Provider):
    platform = "google"

    def __init__(self, model=DEFAULT_MODEL, api_key=None, transport=None,
                 search_enabled=True):
        self._model = model
        self._api_key = api_key or _key_from_environment()
        self._transport = transport or post_json
        self._search_enabled = search_enabled

    @property
    def model(self):
        return self._model

    def credentials_available(self):
        return bool(self._api_key)

    def _params(self, search_ran=None, model_reported=None):
        params = {"web_search": self._cfg("_search_enabled", True),
                  "temperature": "provider_default",
                  "model_requested": self._cfg("_model", DEFAULT_MODEL)}
        if search_ran is not None:
            params["search_ran"] = search_ran
        if model_reported is not None:
            params["model_reported_by_platform"] = model_reported
        return params

    def request_body(self, query_text):
        body = {"contents": [{"role": "user", "parts": [{"text": query_text}]}]}
        if self._search_enabled:
            body["tools"] = [{"google_search": {}}]
        return body

    def _cfg(self, name, default):
        # parse() also runs on a bare instance during verification probing.
        return getattr(self, name, default)

    def parse(self, payload, requested_at, observed_at):
        candidate = _first_candidate(payload)
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if isinstance(p, dict))

        metadata, chunks = _grounding_chunks(candidate)
        # groundingSupports says which chunks actually back a span of the answer;
        # a chunk nobody points at was retrieved but not leaned on.
        supported_indices = set()
        for support in _supports(metadata):
            for index in support.get("groundingChunkIndices") or []:
                if isinstance(index, int):
                    supported_indices.add(index)

        # A URL is cited if ANY of its chunks is supported, not just the first.
        cited_urls = {_web(chunks[i]).get("uri")
                      for i in supported_indices if 0 <= i < len(chunks)}
        # Chunk index -> (is_cited, position within its group). A repeated URL
        # maps to the entry already made for it.
        cited, retrieved, entry_by_url, entry_by_chunk = [], [], {}, {}
        for index, chunk in enumerate(chunks):
            web = _web(chunk)
            url = web.get("uri")
            if not url:
                continue
            if url in entry_by_url:
                entry_by_chunk[index] = entry_by_url[url]
                continue
            # Every grounding uri is a vertexaisearch redirect; the site itself
            # is only in the title. Without this, no clinic's own domain ever
            # matches a Gemini citation and "cited 0 times" is manufactured.
            domain = resolve_domain(url, web.get("title"))
            if url in cited_urls:
                entry_by_url[url] = (True, len(cited))
                cited.append(RawCitation(url=url, title=web.get("title"),
                                         position=len(cited), is_cited=True,
                                         resolved_domain=domain))
            else:
                entry_by_url[url] = (False, len(retrieved))
                retrieved.append(RawCitation(url=url, title=web.get("title"),
                                             position=None, is_cited=False,
                                             resolved_domain=domain))
            entry_by_chunk[index] = entry_by_url[url]

        citations = cited + retrieved
        # Cited sources come first in citations, so a retrieved one sits after them.
        index_by_chunk = {i: (pos if is_cited else len(cited) + pos)
                          for i, (is_cited, pos) in entry_by_chunk.items()}
        spans, dropped = _spans(metadata, text, index_by_chunk,
                                [c.is_cited for c in citations])

        channel_present = "groundingSupports" in metadata
        return ProviderAnswer(
            platform=self.platform,
            model=payload.get("modelVersion") or self._cfg("_model", DEFAULT_MODEL),
            params=self._params(search_ran=bool(chunks),
                                model_reported=bool(payload.get("modelVersion"))),
            requested_at=requested_at,
            observed_at=observed_at,
            answer_text=text,
            citations=citations,
            cited_sources_available=bool(self._cfg("_search_enabled", True) and channel_present),
            # groundingChunks lists every chunk retrieved; groundingSupports
            # says which were leaned on. Both halves visible.
            retrieved_set_available=bool(chunks),
            stop_reason=candidate.get("finishReason"),
            searches=_searches(metadata),
            # Google bills grounded prompts, not searches, and reports no count.
            search_count=None,
            spans=spans,
            spans_dropped=dropped,
            usage=_usage(payload),
            raw_payload=payload,
        )

    def ask(self, query_text):
        if not self._api_key:
            raise ProviderError(f"{' or '.join(API_KEY_ENVS)} is not set")
        requested_at = datetime.now(timezone.utc)
        try:
            payload = self._transport(
                ENDPOINT.format(model=self._model), self.request_body(query_text),
                headers={"x-goog-api-key": self._api_key})
        except HttpError as exc:
            raise ProviderError(f"google request failed: {exc.reason}") from exc
        return self.parse(payload, requested_at, datetime.now(timezone.utc))
