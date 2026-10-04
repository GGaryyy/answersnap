"""Perplexity Sonar API.

Search is not optional on this platform — every answer is grounded — so there is
no search_enabled switch. Parser unverified until a real response is recorded.
"""

import os
from datetime import datetime, timezone

from answersnap.providers.base import Provider, ProviderAnswer, ProviderError, RawCitation, register
from answersnap.providers.http import HttpError, post_json

# Consumer-default tier: the model behind perplexity.ai's default answer.
DEFAULT_MODEL = "sonar"
ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY_ENV = "PERPLEXITY_API_KEY"


@register
class PerplexityProvider(Provider):
    platform = "perplexity"

    def __init__(self, model=DEFAULT_MODEL, api_key=None, transport=None):
        self._model = model
        self._api_key = api_key or os.environ.get(API_KEY_ENV)
        self._transport = transport or post_json

    @property
    def model(self):
        return self._model

    def credentials_available(self):
        return bool(self._api_key)

    def _params(self, model_reported=None):
        params = {"web_search": True, "temperature": "provider_default",
                  "model_requested": self._cfg("_model", DEFAULT_MODEL)}
        if model_reported is not None:
            params["model_reported_by_platform"] = model_reported
        return params

    def request_body(self, query_text):
        return {"model": self._model,
                "messages": [{"role": "user", "content": query_text}]}

    def _cfg(self, name, default):
        # parse() also runs on a bare instance during verification probing.
        return getattr(self, name, default)

    def parse(self, payload, requested_at, observed_at):
        choices = payload.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], dict):
            text = (choices[0].get("message") or {}).get("content") or ""

        # Sonar returns the sources it used for the answer; there is no separate
        # "retrieved but unused" list, so everything here is a citation.
        sources, seen = [], set()
        for entry in payload.get("search_results") or []:
            if isinstance(entry, dict) and entry.get("url"):
                sources.append((entry["url"], entry.get("title")))
        for url in payload.get("citations") or []:
            if isinstance(url, str):
                sources.append((url, None))

        cited = []
        for url, title in sources:
            if url in seen:
                continue
            seen.add(url)
            cited.append(RawCitation(url=url, title=title, position=len(cited),
                                     is_cited=True))

        # The channel is present when either key is there. An empty one is
        # recorded rather than smoothed over: on a platform where every answer
        # is grounded, "no sources at all" is suspicious but not distinguishable
        # from a truncated call, so it is kept as a fact for later.
        channel_present = "citations" in payload or "search_results" in payload
        params = self._params(model_reported=bool(payload.get("model")))
        if channel_present and not cited:
            params = {**params, "empty_citation_channel": True}
        return ProviderAnswer(
            platform=self.platform,
            model=payload.get("model") or self._cfg("_model", DEFAULT_MODEL),
            params=params,
            requested_at=requested_at,
            observed_at=observed_at,
            answer_text=text,
            citations=cited,
            cited_sources_available=channel_present,
            # Sonar returns the sources behind the answer, with no separate
            # "retrieved but unused" list.
            retrieved_set_available=False,
            stop_reason=(choices[0].get("finish_reason") if choices else None),
        )

    def ask(self, query_text):
        if not self._api_key:
            raise ProviderError(f"{API_KEY_ENV} is not set")
        requested_at = datetime.now(timezone.utc)
        try:
            payload = self._transport(
                ENDPOINT, self.request_body(query_text),
                headers={"Authorization": f"Bearer {self._api_key}"})
        except HttpError as exc:
            raise ProviderError(f"perplexity request failed: {exc.reason}") from exc
        return self.parse(payload, requested_at, datetime.now(timezone.utc))
