"""xAI Grok with Live Search (X + web).

The only platform whose answers track live public opinion, which is why it is in
scope at all: smearing and brand hijacking surface on X days to weeks
before they reach the open web. That same liveness makes it the most volatile
platform, so it carries a higher repeat count (§5d).

Citations pointing at X posts are marked `social_post`: they need auth to fetch,
they disappear far more often than web pages, and the account behind them is the
signal that matters for wash campaigns.
"""

import os
from datetime import datetime, timezone
from urllib.parse import urlparse

from answersnap.providers.base import Provider, ProviderAnswer, ProviderError, RawCitation, register
from answersnap.providers.http import HttpError, post_json

DEFAULT_MODEL = "grok-4"
ENDPOINT = "https://api.x.ai/v1/chat/completions"
API_KEY_ENV = "XAI_API_KEY"
SOCIAL_HOSTS = ("x.com", "twitter.com", "www.x.com", "www.twitter.com")


def _source_type(url):
    return "social_post" if (urlparse(url).netloc or "").lower() in SOCIAL_HOSTS else "web"


def _corpus(url):
    # X is a corpus of its own: nothing that watches the open web can see into
    # it, so a poisoning baseline built on web sources says nothing about it.
    return "x" if _source_type(url) == "social_post" else "web"


@register
class XAIProvider(Provider):
    platform = "xai"

    def __init__(self, model=DEFAULT_MODEL, api_key=None, transport=None,
                 search_enabled=True):
        self._model = model
        self._api_key = api_key or os.environ.get(API_KEY_ENV)
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
        body = {"model": self._model,
                "messages": [{"role": "user", "content": query_text}]}
        if self._search_enabled:
            # return_citations is sent explicitly: if it defaults off, the
            # payload carries no citations key at all and Grok's citation data
            # goes permanently silent — on the platform where smearing surfaces
            # first. Unverified until a real response is recorded.
            body["search_parameters"] = {"mode": "on", "return_citations": True,
                                         "sources": [{"type": "web"}, {"type": "x"}]}
        return body

    def _cfg(self, name, default):
        # parse() also runs on a bare instance during verification probing.
        return getattr(self, name, default)

    def parse(self, payload, requested_at, observed_at):
        choices = payload.get("choices") or []
        text = ""
        finish_reason = None
        if choices and isinstance(choices[0], dict):
            text = (choices[0].get("message") or {}).get("content") or ""
            finish_reason = choices[0].get("finish_reason")

        cited, seen = [], set()
        citations = payload.get("citations") or []
        for entry in citations:
            url = entry if isinstance(entry, str) else (entry or {}).get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            cited.append(RawCitation(
                url=url,
                title=(entry.get("title") if isinstance(entry, dict) else None),
                position=len(cited),
                is_cited=True,
                source_type=_source_type(url),
                corpus=_corpus(url),
            ))

        return ProviderAnswer(
            platform=self.platform,
            model=payload.get("model") or self._cfg("_model", DEFAULT_MODEL),
            params=self._params(search_ran="citations" in payload,
                                model_reported=bool(payload.get("model"))),
            requested_at=requested_at,
            observed_at=observed_at,
            answer_text=text,
            citations=cited,
            cited_sources_available=bool(self._cfg("_search_enabled", True)
                                         and "citations" in payload),
            # Live Search returns citations only; what it looked at and passed
            # over is not exposed.
            retrieved_set_available=False,
            stop_reason=finish_reason,
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
            raise ProviderError(f"xai request failed: {exc.reason}") from exc
        return self.parse(payload, requested_at, datetime.now(timezone.utc))
