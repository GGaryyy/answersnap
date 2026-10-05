import json

import pytest
from builders import STAMP, make_config

from answersnap import store
from answersnap.providers import ProviderAnswer, ProviderError, RawCitation


def _answer(**kwargs):
    defaults = dict(platform="anthropic", model="claude-sonnet-5",
                    params={"model_requested": "claude-sonnet-5", "model_reported_by_platform": True},
                    requested_at=STAMP, observed_at=STAMP,
                    answer_text="  Trellis 🦷 raw text\n",
                    citations=[RawCitation(url="https://www.trellis.example/a", is_cited=True,
                                           cited_text="quote", position=0),
                               RawCitation(url="https://x.example/b", is_cited=False)],
                    cited_sources_available=True, retrieved_set_available=True,
                    stop_reason="end_turn")
    return ProviderAnswer(**{**defaults, **kwargs})


def test_answer_record_keeps_raw_text_and_citation_flags():
    config = make_config()
    record = store.answer_record(_answer(), run_id="R", query_index=0,
                                 query=config.prompt_set.queries[0], repeat=1,
                                 prompt_set_version=1, prompt_set_hash="sha256:x")
    assert record["answer_text"] == "  Trellis 🦷 raw text\n"
    assert record["intent"] == "recommendation"
    assert record["citations"][0] == {"url": "https://www.trellis.example/a", "title": None,
                                      "domain": "www.trellis.example", "is_cited": True,
                                      "position": 0, "cited_text": "quote"}
    assert record["citations"][1]["is_cited"] is False
    assert record["cited_sources_available"] is True
    assert record["model_requested"] == "claude-sonnet-5"
    assert record["observed_at"] == STAMP.isoformat()


def test_resolved_domain_is_stored_not_the_redirect_host():
    config = make_config()
    answer = _answer(citations=[RawCitation(
        url="https://vertexaisearch.cloud.google.com/grounding-api-redirect/x",
        resolved_domain="trellis.example")])
    record = store.answer_record(answer, run_id="R", query_index=0,
                                 query=config.prompt_set.queries[0], repeat=0,
                                 prompt_set_version=1, prompt_set_hash="h")
    assert record["citations"][0]["domain"] == "trellis.example"


def test_paths_follow_the_documented_layout(tmp_path):
    assert store.answer_path(tmp_path, "openai", 3, 2) == tmp_path / "answers/openai/q03_r2.json"
    assert store.error_path(tmp_path, "openai", 3, 2).name == "q03_r2.error.json"
    assert store.fetched_path(tmp_path, "https://a.example/x").parent.name == "fetched"
    assert (store.fetched_path(tmp_path, "https://a.example/x")
            != store.fetched_path(tmp_path, "https://a.example/y"))


def test_write_json_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "nested" / "doc.json"
    store.write_json(path, {"k": "值"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": "值"}
    assert [p.name for p in path.parent.iterdir()] == ["doc.json"]


def test_done_pairs_counts_answers_but_not_failures(tmp_path):
    store.write_json(store.answer_path(tmp_path, "anthropic", 0, 0),
                     {"engine": "anthropic", "query_index": 0, "repeat": 0})
    store.write_json(store.error_path(tmp_path, "anthropic", 0, 1),
                     {"engine": "anthropic", "query_index": 0, "repeat": 1})
    assert store.done_pairs(tmp_path) == {("anthropic", 0, 0)}
    assert len(store.load_errors(tmp_path)) == 1


def test_load_answers_records_the_relative_file(tmp_path):
    store.write_json(store.answer_path(tmp_path, "google", 1, 0),
                     {"engine": "google", "query_index": 1, "repeat": 0})
    assert store.load_answers(tmp_path)[0]["_file"] == "answers/google/q01_r0.json"


def test_error_record_names_the_failure_without_payloads():
    record = store.error_record(run_id="R", engine="openai", query_index=0, repeat=0,
                                requested_at=STAMP, error=ProviderError("openai request failed: RateLimitError"))
    assert record["error_type"] == "ProviderError"
    assert record["message"] == "openai request failed: RateLimitError"


def test_read_manifest_on_a_non_run_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="not an answersnap run"):
        store.read_manifest(tmp_path)


def _seed_run(out_root, run_id, *, version, prompt_hash, mode="live"):
    run_dir = store.brand_dir(out_root, "Trellis") / run_id
    store.write_manifest(run_dir, {"run_id": run_id, "mode": mode,
                                   "prompt_set": {"version": version, "hash": prompt_hash}})


def test_freeze_guard_refuses_edited_questions_at_the_same_version(tmp_path):
    _seed_run(tmp_path, "20260101T000000Z", version=1, prompt_hash="sha256:old")
    with pytest.raises(store.PromptSetChanged, match="Bump prompt_set.version"):
        store.refuse_if_prompt_set_changed(tmp_path, "Trellis", 1, "sha256:new")


def test_freeze_guard_allows_same_questions_and_new_versions(tmp_path):
    _seed_run(tmp_path, "20260101T000000Z", version=1, prompt_hash="sha256:old")
    store.refuse_if_prompt_set_changed(tmp_path, "Trellis", 1, "sha256:old")
    store.refuse_if_prompt_set_changed(tmp_path, "Trellis", 2, "sha256:new")


def test_freeze_guard_ignores_dry_runs_and_other_brands(tmp_path):
    _seed_run(tmp_path, "20260101T000000Z", version=1, prompt_hash="sha256:old", mode="dry_run")
    store.refuse_if_prompt_set_changed(tmp_path, "Trellis", 1, "sha256:new")
    store.refuse_if_prompt_set_changed(tmp_path, "Other Brand", 1, "sha256:new")


def test_runs_started_in_the_same_second_get_their_own_directories(tmp_path):
    first, first_id = store.new_run_dir(tmp_path, "Trellis", now=STAMP)
    first.mkdir(parents=True)
    second, second_id = store.new_run_dir(tmp_path, "Trellis", now=STAMP)
    assert first != second and second_id == f"{first_id}-2"
    assert not second.exists()


# ---------------------------------------------------------------- record-2
def test_only_answer_files_are_loaded_as_answers(tmp_path):
    from builders import record
    store.write_json(store.answer_path(tmp_path, "openai", 0, 0), record(engine="openai"))
    store.write_raw_payload(store.raw_path(tmp_path, "openai", 0, 0), {"output": []})
    (tmp_path / "answers/openai/q00_r1.json.tmp").write_text("{half")
    (tmp_path / "answers/openai/notes.json").write_text("{}")
    assert [r["_file"] for r in store.load_answers(tmp_path)] == ["answers/openai/q00_r0.json"]


def test_an_answer_1_record_loads_with_every_new_field_unknown(tmp_path):
    from builders import record
    legacy = record(legacy=True)
    legacy.pop("_file")
    store.write_json(store.answer_path(tmp_path, "anthropic", 0, 0), legacy)
    (loaded,) = store.load_answers(tmp_path)
    assert loaded["schema"] == "answer-1"
    for key in store.ANSWER_2_DEFAULTS:
        assert loaded[key] is None, key


def test_raw_payload_digest_is_of_the_bytes_on_disk(tmp_path):
    import hashlib
    path = store.raw_path(tmp_path, "google", 3, 1)
    written = store.write_raw_payload(path, {"b": 1, "a": "值"})
    assert path.name == "q03_r1.raw.json"
    data = path.read_bytes()
    assert written == {"sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert "值" in data.decode("utf-8")


def test_a_hundredth_question_is_still_an_answer(tmp_path):
    from builders import record
    path = store.answer_path(tmp_path, "openai", 100, 0)
    assert path.name == "q100_r0.json"
    store.write_json(path, record(engine="openai", query_index=100))
    assert [r["query_index"] for r in store.load_answers(tmp_path)] == [100]
