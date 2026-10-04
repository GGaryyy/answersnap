"""Provider adapter contract.

Every adapter must obey the measurement rules (README, "Measurement rules"):

  - the query string is sent verbatim and identically across platforms
  - no system prompt
  - a fresh conversation per call, no history
  - the provider's default temperature — never 0, because consumers don't get 0
  - the consumer-default model tier, not the flagship
  - direct API only; never an agent harness or a consumer subscription

Adapters return raw material. They do not count, judge or normalise anything.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# A parser that has never seen a real response from its platform is not merely
# unpolished: when it gets citations wrong, the data says "this platform cites
# nothing", which is indistinguishable from suppression or poisoning. So the
# verified flag is derived from a recorded response existing on disk, never
# declared by hand.
# Absolute and inside the package: a relative path would make the verification
# verdict depend on the process's working directory, and a path under tests/
# would be stripped out of the container image — so every platform would report
# unverified in production and the gate would degrade into a permanent override.
RECORDED_DIR = Path(__file__).resolve().parent / "recorded"


@dataclass(frozen=True)
class RawCitation:
    url: str
    title: str | None = None
    # position is the order in the ANSWER, which is what "how much the model
    # leaned on this source" means. Search-result order is not that, so a
    # retrieved-but-uncited source carries no position.
    position: int | None = None
    is_cited: bool = True
    cited_text: str | None = None
    source_type: str = "web"
    corpus: str = "web"
    source_meta: dict | None = None
    # Set by adapters whose platform hides the real site behind a redirect
    # (Gemini's grounding links). Left None where the URL is the site.
    resolved_domain: str | None = None

    @property
    def domain(self):
        return self.resolved_domain or (urlparse(self.url).netloc or "").lower()


@dataclass(frozen=True)
class ProviderAnswer:
    platform: str
    model: str
    params: dict
    requested_at: datetime
    observed_at: datetime
    answer_text: str  # raw; never trimmed or normalised
    citations: list[RawCitation] = field(default_factory=list)
    # False when this particular call could not have told us which sources were
    # cited. Downstream this must suppress citation-rate statistics rather than
    # read as "nothing was cited".
    cited_sources_available: bool = True
    # False when the platform never shows which sources it retrieved without
    # citing. Downstream this must suppress the cited/retrieved ratio rather
    # than read as a perfect one.
    retrieved_set_available: bool = False
    stop_reason: str | None = None


class ProviderError(RuntimeError):
    pass


_VERIFIED_CACHE = {}


class Provider(ABC):
    platform: str

    @property
    @abstractmethod
    def model(self):
        ...

    @abstractmethod
    def ask(self, query_text):
        ...

    def credentials_available(self):
        """Whether this adapter could make a call at all.

        Asked before sampling, because a missing key otherwise shows up as every
        request failing — which on a dashboard is indistinguishable from the
        platform going dark on us.
        """
        return True

    @classmethod
    def recorded_response_path(cls):
        return RECORDED_DIR / f"{cls.platform}.json"

    @classmethod
    def response_shape_verified(cls):
        """True only if a real recorded response actually parses.

        Checking that a file exists would measure the wrong thing: an empty stub
        would mark the platform verified while proving nothing. The claim is
        "this parser has handled a real response", so that is what gets tested.
        """
        if cls.platform in _VERIFIED_CACHE:
            return _VERIFIED_CACHE[cls.platform]
        verdict = cls._probe_recorded_response()
        _VERIFIED_CACHE[cls.platform] = verdict
        return verdict

    @classmethod
    def _probe_recorded_response(cls):
        path = cls.recorded_response_path()
        if not path.exists():
            return False
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        payload = recorded.get("payload")
        if not isinstance(payload, dict):
            return False
        stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        try:
            answer = cls.__new__(cls).parse(payload, stamp, stamp)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        # At least one CITED source, not merely retrieved ones. A recording with
        # 14 retrieved-only sources and zero citations is exactly the failure
        # this gate exists to catch, and it would sail through a check that only
        # counted rows.
        return bool(answer.answer_text
                    and any(c.is_cited for c in answer.citations))

    @classmethod
    def reset_verification_cache(cls):
        _VERIFIED_CACHE.pop(cls.platform, None)

    @classmethod
    def load_recorded_response(cls):
        path = cls.recorded_response_path()
        if not path.exists():
            raise ProviderError(f"no recorded response for '{cls.platform}'; "
                                "record a real response for this platform first")
        return json.loads(path.read_text(encoding="utf-8"))

    @abstractmethod
    def parse(self, payload, requested_at, observed_at):
        """Turn a raw response payload into a ProviderAnswer.

        Kept separate from `ask` so the parser can be tested against a recorded
        real response instead of against a fixture we invented ourselves.
        """


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.platform] = cls
    return cls


def get_provider(platform, **kwargs):
    if platform not in _REGISTRY:
        raise ProviderError(f"no adapter registered for platform '{platform}'")
    return _REGISTRY[platform](**kwargs)


def get_provider_class(platform):
    if platform not in _REGISTRY:
        raise ProviderError(f"no adapter registered for platform '{platform}'")
    return _REGISTRY[platform]


def registered_platforms():
    return sorted(_REGISTRY)


def verified_platforms():
    return sorted(p for p, cls in _REGISTRY.items() if cls.response_shape_verified())
