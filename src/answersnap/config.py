"""The one input file: who the brand is, and the frozen questions to ask about it.

The prompt set is the measuring stick. Rates are only comparable while the
questions stay identical, so a prompt set is versioned, and every snapshot
records the version and a hash of the exact questions it was answered against.
Editing a question means a new version — see store.refuse_if_prompt_set_changed.
"""

import hashlib
import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from answersnap.providers.citation_host import strip_www

# ---------------------------------------------------------------- constants
SCHEMA_VERSION = 1
# Engines with a parser that has handled a real recorded response. Adapters for
# other platforms exist but stay out until they are verified the same way.
SUPPORTED_ENGINES = ("anthropic", "openai", "google")
DEFAULT_REPEATS = 3
DEFAULT_RECOMMENDATION_INTENTS = ("recommendation",)
MAX_REPEATS = 50
YAML_SUFFIXES = (".yaml", ".yml")


class ConfigError(ValueError):
    """The config file cannot be used as written; the message says where."""


class _Strict(BaseModel):
    # A misspelt key ("alias" for "aliases") would otherwise be dropped in
    # silence, and a missing alias looks exactly like a brand AI never names.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Competitor(_Strict):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)

    @field_validator("aliases")
    @classmethod
    def _no_blank_aliases(cls, aliases):
        cleaned = [alias.strip() for alias in aliases]
        if any(not alias for alias in cleaned):
            raise ValueError("aliases must not be blank")
        return cleaned


class Brand(Competitor):
    owned_domains: list[str] = Field(default_factory=list)

    @field_validator("owned_domains")
    @classmethod
    def _normalise_domains(cls, domains):
        # Matched against each citation's resolved host, which carries no
        # scheme, path or www — so the config must not either.
        cleaned = []
        for domain in domains:
            host = re.sub(r"^[a-z]+://", "", domain.strip().lower()).split("/")[0].split(":")[0]
            if not host:
                raise ValueError(f"{domain!r} is not a domain")
            cleaned.append(strip_www(host))
        return cleaned


class Query(_Strict):
    text: str = Field(min_length=1)
    intent: str = Field(min_length=1)


class PromptSet(_Strict):
    version: int = Field(ge=1)
    queries: list[Query] = Field(min_length=1)

    @field_validator("queries")
    @classmethod
    def _unique_texts(cls, queries):
        # Two identical questions would be one question sampled twice as often,
        # quietly weighting the rates toward it.
        seen = set()
        for query in queries:
            if query.text in seen:
                raise ValueError(f"duplicate query text: {query.text!r}")
            seen.add(query.text)
        return queries


class Config(_Strict):
    schema_: int = Field(alias="schema")
    brand: Brand
    competitors: list[Competitor] = Field(default_factory=list)
    engines: list[str] = Field(default_factory=lambda: list(SUPPORTED_ENGINES), min_length=1)
    repeats: int = Field(default=DEFAULT_REPEATS, ge=1, le=MAX_REPEATS)
    recommendation_intents: list[str] = Field(
        default_factory=lambda: list(DEFAULT_RECOMMENDATION_INTENTS))
    prompt_set: PromptSet

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("schema_")
    @classmethod
    def _known_schema(cls, value):
        if value != SCHEMA_VERSION:
            raise ValueError(f"schema {value} is not supported; expected {SCHEMA_VERSION}")
        return value

    @field_validator("engines")
    @classmethod
    def _known_engines(cls, engines):
        unknown = [e for e in engines if e not in SUPPORTED_ENGINES]
        if unknown:
            raise ValueError(f"unsupported engine(s) {unknown}; "
                             f"supported: {', '.join(SUPPORTED_ENGINES)}")
        if len(set(engines)) != len(engines):
            raise ValueError("engines must not repeat")
        return engines

    @model_validator(mode="after")
    def _entities_distinct(self):
        names = [self.brand.name, *(c.name for c in self.competitors)]
        if len(set(names)) != len(names):
            raise ValueError("brand and competitor names must all differ")
        return self

    def recommendation_queries(self):
        return [i for i, q in enumerate(self.prompt_set.queries)
                if q.intent in self.recommendation_intents]

    def warnings(self):
        notes = []
        if not self.recommendation_queries():
            notes.append(
                "no query has an intent listed in recommendation_intents "
                f"({', '.join(self.recommendation_intents)}); "
                "'named in recommendation answers' will be reported as not measured")
        if not self.brand.owned_domains:
            notes.append("brand.owned_domains is empty; citation rate will be not measured")
        return notes

    def to_dict(self):
        return self.model_dump(mode="json", by_alias=True)


def _canonical(document):
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_set_hash(prompt_set, recommendation_intents=None):
    """Identity of the exact questions asked and of which ones count as
    recommendation questions. Key order and whitespace in the file do not
    change it; changing a question's text or intent, or the intents that the
    headline counts over, does — either one changes what the headline means."""
    document = {"version": prompt_set.version,
                "queries": [{"text": q.text, "intent": q.intent} for q in prompt_set.queries],
                "recommendation_intents": sorted(recommendation_intents or [])}
    return "sha256:" + hashlib.sha256(_canonical(document).encode("utf-8")).hexdigest()


def prompt_identity(config):
    """The hash every snapshot and the freeze guard use for this config."""
    return prompt_set_hash(config.prompt_set, config.recommendation_intents)


def config_hash(config):
    return "sha256:" + hashlib.sha256(_canonical(config.to_dict()).encode("utf-8")).hexdigest()


def brand_slug(name):
    slug = re.sub(r"[^\w]+", "-", name.casefold()).strip("-")
    return slug or "brand"


def _read_document(path):
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in YAML_SUFFIXES:
            return yaml.safe_load(text)
        return json.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: not valid {path.suffix.lstrip('.') or 'JSON'}: {exc}") from exc


def _describe(error):
    lines = []
    for issue in error.errors():
        where = ".".join(str(part) for part in issue["loc"]) or "(top level)"
        lines.append(f"  {where}: {issue['msg']}")
    return "\n".join(lines)


def parse_config(document, source="config"):
    if not isinstance(document, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    try:
        return Config.model_validate(document)
    except ValidationError as exc:
        raise ConfigError(f"{source} is not valid:\n{_describe(exc)}") from exc


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no config file at {path}; create one with `answersnap init`")
    return parse_config(_read_document(path), source=str(path))
