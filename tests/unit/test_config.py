import json

import pytest
from builders import BASE_CONFIG, config_dict, make_config

from answersnap.config import (
    ConfigError,
    brand_slug,
    config_hash,
    load_config,
    parse_config,
    prompt_set_hash,
)


def test_a_valid_config_loads_with_defaults_applied():
    document = config_dict()
    del document["repeats"]
    config = parse_config(document)
    assert config.repeats == 3
    assert config.brand.owned_domains == ["trellis.example"]
    assert config.recommendation_queries() == [0]


def test_yaml_and_json_files_load_to_the_same_config(tmp_path):
    import yaml

    yaml_path = tmp_path / "c.yaml"
    json_path = tmp_path / "c.json"
    yaml_path.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    json_path.write_text(json.dumps(BASE_CONFIG), encoding="utf-8")
    assert load_config(yaml_path) == load_config(json_path)


def test_missing_file_points_at_init(tmp_path):
    with pytest.raises(ConfigError, match="answersnap init"):
        load_config(tmp_path / "absent.yaml")


def test_broken_yaml_is_a_config_error_not_a_traceback(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("brand: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid yaml"):
        load_config(path)


def test_misspelt_key_is_rejected_with_its_path():
    document = config_dict(brand={"name": "Trellis", "alias": ["T"]})
    with pytest.raises(ConfigError, match=r"brand\.alias"):
        parse_config(document)


@pytest.mark.parametrize("prompt_set, expected", [
    ({"version": 1, "queries": []}, "prompt_set.queries"),
    ({"version": 0, "queries": [{"text": "q", "intent": "i"}]}, "prompt_set.version"),
    ({"version": 1, "queries": [{"text": "q", "intent": "i"}, {"text": "q", "intent": "j"}]},
     "duplicate query text"),
])
def test_unusable_prompt_sets_are_refused(prompt_set, expected):
    with pytest.raises(ConfigError, match=expected):
        parse_config(config_dict(prompt_set=prompt_set))


def test_unknown_or_repeated_engines_are_refused():
    with pytest.raises(ConfigError, match="unsupported engine"):
        parse_config(config_dict(engines=["anthropic", "perplexity"]))
    with pytest.raises(ConfigError, match="must not repeat"):
        parse_config(config_dict(engines=["openai", "openai"]))


def test_unsupported_schema_is_refused():
    with pytest.raises(ConfigError, match="schema 2 is not supported"):
        parse_config(config_dict(schema=2))


def test_brand_and_competitor_names_must_differ():
    with pytest.raises(ConfigError, match="must all differ"):
        parse_config(config_dict(competitors=[{"name": "Trellis"}]))


def test_blank_alias_is_refused():
    with pytest.raises(ConfigError, match="must not be blank"):
        parse_config(config_dict(brand={"name": "Trellis", "aliases": ["  "]}))


@pytest.mark.parametrize("raw, expected", [
    ("https://www.trellis.example/about", "trellis.example"),
    ("WWW.Trellis.Example", "trellis.example"),
    ("wallace.example", "wallace.example"),  # strip_www must not eat a leading "w"
])
def test_owned_domains_are_normalised_to_bare_hosts(raw, expected):
    config = make_config(brand={"name": "Trellis", "owned_domains": [raw]})
    assert config.brand.owned_domains == [expected]


def test_prompt_hash_ignores_key_order_but_not_wording():
    first = make_config()
    reordered = parse_config(json.loads(json.dumps(config_dict()), object_pairs_hook=lambda p: dict(reversed(p))))
    assert prompt_set_hash(first.prompt_set) == prompt_set_hash(reordered.prompt_set)

    edited = config_dict()
    edited["prompt_set"] = {"version": 1, "queries": [
        {"text": "Best clinic for implants??", "intent": "recommendation"},
        {"text": "How long do implants last?", "intent": "how_to"}]}
    assert prompt_set_hash(make_config(**edited).prompt_set) != prompt_set_hash(first.prompt_set)


def test_prompt_hash_changes_with_intent_and_version():
    base = make_config()
    retagged = config_dict(prompt_set={"version": 1, "queries": [
        {"text": "Best clinic for implants?", "intent": "trust"},
        {"text": "How long do implants last?", "intent": "how_to"}]})
    bumped = config_dict(prompt_set={**BASE_CONFIG["prompt_set"], "version": 2})
    hashes = {prompt_set_hash(c.prompt_set) for c in (base, make_config(**retagged), make_config(**bumped))}
    assert len(hashes) == 3


def test_config_hash_tracks_any_change():
    assert config_hash(make_config()) == config_hash(make_config())
    assert config_hash(make_config()) != config_hash(make_config(repeats=5))


def test_warnings_name_what_will_not_be_measured():
    config = make_config(recommendation_intents=["nothing_matches"],
                         brand={"name": "Trellis"})
    notes = config.warnings()
    assert any("not measured" in n and "recommendation" in n for n in notes)
    assert any("owned_domains is empty" in n for n in notes)
    assert make_config().warnings() == []


@pytest.mark.parametrize("name, slug", [
    ("Example Coffee Co.", "example-coffee-co"),
    ("示範牙醫", "示範牙醫"),
    ("!!!", "brand"),
])
def test_brand_slug(name, slug):
    assert brand_slug(name) == slug


def test_owned_domains_drop_ports():
    config = make_config(brand={"name": "Trellis", "owned_domains": ["https://trellis.example:443/x"]})
    assert config.brand.owned_domains == ["trellis.example"]


def test_prompt_identity_changes_when_the_counted_intents_change():
    from answersnap.config import prompt_identity
    assert prompt_identity(make_config()) != prompt_identity(
        make_config(recommendation_intents=["recommendation", "trust"]))
