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
    register,
)
from answersnap.providers.citation_host import resolve_domain
from answersnap.providers.http import HttpError, post_json

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
        for support in metadata.get("groundingSupports") or []:
            if isinstance(support, dict):
                for index in support.get("groundingChunkIndices") or []:
                    if isinstance(index, int):
                        supported_indices.add(index)

        cited, retrieved, seen = [], [], set()
        for index, chunk in enumerate(chunks):
            web = chunk.get("web") or {}
            url = web.get("uri")
            if not url or url in seen:
                continue
            seen.add(url)
            # Every grounding uri is a vertexaisearch redirect; the site itself
            # is only in the title. Without this, no clinic's own domain ever
            # matches a Gemini citation and "cited 0 times" is manufactured.
            domain = resolve_domain(url, web.get("title"))
            if index in supported_indices:
                cited.append(RawCitation(url=url, title=web.get("title"),
                                         position=len(cited), is_cited=True,
                                         resolved_domain=domain))
            else:
                retrieved.append(RawCitation(url=url, title=web.get("title"),
                                             position=None, is_cited=False,
                                             resolved_domain=domain))

        channel_present = "groundingSupports" in metadata
        return ProviderAnswer(
            platform=self.platform,
            model=payload.get("modelVersion") or self._cfg("_model", DEFAULT_MODEL),
            params=self._params(search_ran=bool(chunks),
                                model_reported=bool(payload.get("modelVersion"))),
            requested_at=requested_at,
            observed_at=observed_at,
            answer_text=text,
            citations=cited + retrieved,
            cited_sources_available=bool(self._cfg("_search_enabled", True) and channel_present),
            # groundingChunks lists every chunk retrieved; groundingSupports
            # says which were leaned on. Both halves visible.
            retrieved_set_available=bool(chunks),
            stop_reason=candidate.get("finishReason"),
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
