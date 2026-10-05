"""The sampling loop with stand-in engines: what gets asked, written and counted."""

from builders import STAMP, make_config

from answersnap import run, store
from answersnap.providers import ProviderAnswer, ProviderError


class ScriptedEngine:
    """Answers from a script; raises ProviderError for listed (query, repeat)."""

    mode = "live"

    def __init__(self, name, fail=(), model="model-x", **extra):
        self.name = name
        self.model = model
        self.asked = []
        self._fail = set(fail)
        self._extra = extra

    def answer(self, query_text, repeat):
        self.asked.append((query_text, repeat))
        if (query_text, repeat) in self._fail:
            raise ProviderError(f"{self.name} request failed: RateLimitError")
        return ProviderAnswer(platform=self.name, model=self.model,
                              params={"model_requested": self.model}, requested_at=STAMP,
                              observed_at=STAMP, answer_text=f"Trellis says {self.name}",
                              cited_sources_available=True, **self._extra)


def _start(tmp_path, config, engines, skipped=None, max_calls=None, done=frozenset(),
           keep_raw=True, mode="live"):
    tasks = run.plan_tasks(config, list(engines), config.repeats, done)
    if max_calls is not None:
        tasks = tasks[:max_calls]
    manifest = run.initial_manifest(config, run_id="R1", mode=mode, engines=engines,
                                    skipped=skipped or {}, tasks=tasks, repeats=config.repeats,
                                    max_calls=max_calls, started_at=STAMP,
                                    cost=run.estimate_cost(tasks), keep_raw=keep_raw)
    run.execute(config, tmp_path, manifest, engines, tasks, workers=1, clock=lambda: STAMP)
    return run.finalise(manifest, tmp_path, STAMP), tasks


def test_plan_is_repeat_major_and_skips_done_pairs():
    config = make_config()
    tasks = run.plan_tasks(config, ["anthropic", "openai"], 2, done={("openai", 0, 0)})
    assert [(t.engine, t.query_index, t.repeat) for t in tasks[:3]] == [
        ("anthropic", 0, 0), ("anthropic", 1, 0), ("openai", 1, 0)]
    assert len(tasks) == 2 * 2 * 2 - 1


def test_estimate_prices_typical_usage_and_refuses_unpriced_models():
    config = make_config()
    tasks = run.plan_tasks(config, ["anthropic", "openai"], 2)
    # anthropic: (17599 * $2 + 1949 * $10) / 1M + 2 searches * $0.01 = $0.0747 a call.
    # openai: (7464 * $1.25 + 1700 * $10) / 1M + 1 search * $0.01 = $0.0363 a call.
    assert run.estimate_cost(tasks) == {"anthropic": (4, 0.30), "openai": (4, 0.15)}
    assert run.estimate_cost(tasks, {"openai": "gpt-unpriced"})["openai"] == (4, None)


def test_every_question_is_sent_verbatim_and_written(tmp_path):
    config = make_config()
    engines = {"anthropic": ScriptedEngine("anthropic"), "openai": ScriptedEngine("openai")}
    manifest, _ = _start(tmp_path, config, engines)
    texts = [q.text for q in config.prompt_set.queries]
    assert sorted(engines["anthropic"].asked) == sorted((t, r) for t in texts for r in range(2))
    assert len(store.load_answers(tmp_path)) == 8
    assert manifest["status"] == run.STATUS_COMPLETE
    entry = manifest["engines"]["anthropic"]
    assert {k: entry[k] for k in ("status", "reason", "model_requested", "models_reported",
                                  "planned", "done", "failed")} == {
        "status": "ok", "reason": None, "model_requested": "model-x",
        "models_reported": ["model-x"], "planned": 4, "done": 4, "failed": 0}


def test_failed_calls_are_kept_and_make_the_run_incomplete(tmp_path):
    config = make_config()
    first = config.prompt_set.queries[0].text
    engines = {"anthropic": ScriptedEngine("anthropic", fail={(first, 1)}),
               "openai": ScriptedEngine("openai")}
    manifest, _ = _start(tmp_path, config, engines)
    errors = store.load_errors(tmp_path)
    assert [(e["engine"], e["query_index"], e["repeat"]) for e in errors] == [("anthropic", 0, 1)]
    assert manifest["engines"]["anthropic"]["status"] == "partial"
    assert manifest["calls"]["failed"] == 1
    assert manifest["status"] == run.STATUS_INCOMPLETE


def test_max_calls_cap_leaves_a_visibly_partial_run(tmp_path):
    config = make_config()
    engines = {"anthropic": ScriptedEngine("anthropic"), "openai": ScriptedEngine("openai")}
    manifest, tasks = _start(tmp_path, config, engines, max_calls=3)
    assert len(tasks) == 3 and manifest["calls"]["made"] == 3
    assert manifest["status"] == run.STATUS_INCOMPLETE


def test_skipped_engine_is_named_and_makes_the_run_incomplete(tmp_path):
    config = make_config()
    engines = {"anthropic": ScriptedEngine("anthropic")}
    skipped = {"openai": ("skipped_no_credentials", "ChatGPT: not run — OPENAI_API_KEY is not set")}
    manifest, _ = _start(tmp_path, config, engines, skipped=skipped)
    assert manifest["engines"]["openai"]["status"] == "skipped_no_credentials"
    assert "OPENAI_API_KEY" in manifest["engines"]["openai"]["reason"]
    assert manifest["status"] == run.STATUS_INCOMPLETE


def test_engine_left_out_on_purpose_is_a_subset_not_complete(tmp_path):
    config = make_config()
    manifest, _ = _start(tmp_path, config, {"anthropic": ScriptedEngine("anthropic")})
    assert manifest["engines"]["openai"]["status"] == run.SKIPPED_NOT_REQUESTED
    assert manifest["status"] == run.STATUS_COMPLETE_SUBSET


def test_resume_retries_only_what_is_missing_and_clears_old_errors(tmp_path):
    config = make_config()
    first = config.prompt_set.queries[0].text
    _start(tmp_path, config, {"anthropic": ScriptedEngine("anthropic", fail={(first, 0)})})
    assert store.load_errors(tmp_path)

    retry = ScriptedEngine("anthropic")
    manifest, tasks = _start(tmp_path, config, {"anthropic": retry},
                             done=store.done_pairs(tmp_path))
    assert retry.asked == [(first, 0)]
    assert store.load_errors(tmp_path) == []
    assert manifest["engines"]["anthropic"]["done"] == 4
    assert manifest["engines"]["anthropic"]["status"] == "ok"
    assert manifest["status"] == run.STATUS_COMPLETE_SUBSET


def test_parallel_workers_write_the_same_files(tmp_path):
    config = make_config()
    engines = {"anthropic": ScriptedEngine("anthropic"), "openai": ScriptedEngine("openai")}
    tasks = run.plan_tasks(config, list(engines), config.repeats)
    manifest = run.initial_manifest(config, run_id="R", mode="live", engines=engines, skipped={},
                                    tasks=tasks, repeats=config.repeats, max_calls=None,
                                    started_at=STAMP, cost={})
    run.execute(config, tmp_path, manifest, engines, tasks, workers=4, clock=lambda: STAMP)
    assert manifest["calls"]["made"] == 8
    assert len(store.done_pairs(tmp_path)) == 8


# ---------------------------------------------------------------- review regressions
class ShapeChangedEngine(ScriptedEngine):
    def answer(self, query_text, repeat):
        self.asked.append((query_text, repeat))
        raise KeyError("candidates")   # what a parser raises when a response shape changes


def test_a_parser_error_is_recorded_like_any_failed_call(tmp_path):
    config = make_config()
    manifest, _ = _start(tmp_path, config, {"anthropic": ShapeChangedEngine("anthropic")})
    errors = store.load_errors(tmp_path)
    assert len(errors) == 4 and {e["error_type"] for e in errors} == {"KeyError"}
    assert manifest["status"] == run.STATUS_INCOMPLETE


def test_interrupting_a_parallel_run_cancels_queued_calls(tmp_path):
    import time

    import pytest

    class InterruptingEngine(ScriptedEngine):
        """The first call is Ctrl-C; every other call is slow, so a pool that
        ignored the interrupt would go on to make all of them."""

        def answer(self, query_text, repeat):
            if not self.asked:
                self.asked.append((query_text, repeat))
                raise KeyboardInterrupt
            time.sleep(0.05)
            return super().answer(query_text, repeat)

    config = make_config(repeats=20)
    engine = InterruptingEngine("anthropic")
    tasks = run.plan_tasks(config, ["anthropic"], 20)
    manifest = run.initial_manifest(config, run_id="R", mode="live", engines={"anthropic": engine},
                                    skipped={}, tasks=tasks, repeats=20, max_calls=None,
                                    started_at=STAMP, cost={})
    with pytest.raises(KeyboardInterrupt):
        run.execute(config, tmp_path, manifest, {"anthropic": engine}, tasks, workers=2,
                    clock=lambda: STAMP)
    assert len(engine.asked) < len(tasks) / 2
    assert store.read_manifest(tmp_path)["status"] == run.STATUS_RUNNING


def test_settle_recounts_a_run_that_never_finished(tmp_path):
    config = make_config()
    _start(tmp_path, config, {"anthropic": ScriptedEngine("anthropic"),
                              "openai": ScriptedEngine("openai")}, max_calls=3)
    manifest = store.read_manifest(tmp_path)
    manifest.update(status=run.STATUS_RUNNING, calls={"planned": 99, "made": 0, "failed": 0,
                                                      "cap": None, "this_session": 0})
    run.settle(manifest, tmp_path)
    assert manifest["calls"]["made"] == 3 and manifest["status"] == run.STATUS_INCOMPLETE


# ---------------------------------------------------------------- record-2
USAGE = {"input_tokens": 1000, "output_tokens": 200, "reasoning_tokens": 50,
         "total_tokens": None, "output_includes_reasoning": True, "raw": {}}


def _priced_engine(**extra):
    return ScriptedEngine("anthropic", model="claude-sonnet-5", usage=USAGE, search_count=3,
                          **extra)


def test_raw_payload_is_written_beside_the_record_with_a_matching_digest(tmp_path):
    import hashlib
    config = make_config()
    _start(tmp_path, config, {"anthropic": _priced_engine(raw_payload={"k": "值"})})
    record = store.load_answers(tmp_path)[0]
    assert record["schema"] == "answer-2" and record["raw_status"] == "kept"
    data = (tmp_path / record["raw_file"]).read_bytes()
    assert record["raw_sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
    assert record["raw_bytes"] == len(data)
    assert store.read_json(tmp_path / record["raw_file"]) == {"k": "值"}
    # The raw file is not mistaken for an answer.
    assert len(store.load_answers(tmp_path)) == 4


def test_no_raw_keeps_no_file_and_says_so(tmp_path):
    config = make_config()
    manifest, _ = _start(tmp_path, config, {"anthropic": _priced_engine(raw_payload={"k": 1})},
                         keep_raw=False)
    assert manifest["raw_payloads"] == "not_kept"
    assert not list(tmp_path.glob("answers/*/*.raw.json"))
    assert {r["raw_status"] for r in store.load_answers(tmp_path)} == {"not_kept"}


def test_engine_without_a_payload_records_not_provided(tmp_path):
    _start(tmp_path, make_config(), {"anthropic": ScriptedEngine("anthropic")})
    assert {r["raw_status"] for r in store.load_answers(tmp_path)} == {"not_provided"}


def test_cost_is_estimated_per_answer_at_write_time(tmp_path):
    _start(tmp_path, make_config(), {"anthropic": _priced_engine()})
    record = store.load_answers(tmp_path)[0]
    # (1000 * $2 + 200 * $10) / 1M + 3 * $0.01
    assert record["cost_estimate_usd"] == 0.034
    assert record["cost_basis"]["kind"] == "list_price"
    assert record["cost_basis"]["searches_priced"] is True


def test_dry_run_and_unpriced_models_get_a_reason_not_a_number(tmp_path):
    _start(tmp_path / "dry", make_config(), {"anthropic": _priced_engine()}, mode="dry_run")
    _start(tmp_path / "live", make_config(), {"anthropic": ScriptedEngine("anthropic")})
    dry = store.load_answers(tmp_path / "dry")[0]
    unpriced = store.load_answers(tmp_path / "live")[0]
    assert dry["cost_estimate_usd"] is None and dry["cost_basis"]["reason"] == "dry_run"
    assert unpriced["cost_basis"] == {"kind": "none", "reason": "no_list_price"}


def test_settle_totals_usage_and_cost_per_engine_only(tmp_path):
    engines = {"anthropic": _priced_engine(), "openai": ScriptedEngine("openai")}
    manifest, _ = _start(tmp_path, make_config(), engines)
    claude = manifest["engines"]["anthropic"]
    assert claude["usage"] == {"answers": 4, "with_usage": 4, "input_tokens": 4000,
                               "output_tokens": 800, "reasoning_tokens": 200,
                               "search_count": 12, "answers_with_search_count": 4}
    assert claude["cost_from_usage_usd"]["usd"] == 0.136
    gpt = manifest["engines"]["openai"]
    # Nothing reported reads as unknown, not zero.
    assert gpt["usage"]["input_tokens"] is None and gpt["usage"]["search_count"] is None
    assert gpt["cost_from_usage_usd"] == {"usd": None, "answers_priced": 0,
                                          "answers_unpriced": 4, "basis": None}
    assert "usage" not in manifest["calls"]


def test_resuming_an_answer_1_run_leaves_old_records_and_reports_both_schemas(tmp_path):
    from builders import record
    config = make_config()
    legacy = record(query_index=0, repeat=0, legacy=True)
    legacy.pop("_file")
    store.write_json(store.answer_path(tmp_path, "anthropic", 0, 0), legacy)
    manifest, tasks = _start(tmp_path, config, {"anthropic": ScriptedEngine("anthropic")},
                             done=store.done_pairs(tmp_path))
    assert len(tasks) == 3
    assert store.read_json(store.answer_path(tmp_path, "anthropic", 0, 0)) == legacy
    assert manifest["record_schemas_on_disk"] == ["answer-1", "answer-2"]


def test_resume_with_a_different_raw_setting_reports_mixed(tmp_path):
    config = make_config()
    first = config.prompt_set.queries[0].text
    engine = _priced_engine(raw_payload={"k": 1}, fail={(first, 0)})
    _start(tmp_path, config, {"anthropic": engine})
    retry = _priced_engine(raw_payload={"k": 2})
    manifest, tasks = _start(tmp_path, config, {"anthropic": retry},
                             done=store.done_pairs(tmp_path), keep_raw=False)
    assert len(tasks) == 1
    assert manifest["raw_payloads"] == "not_kept"
    assert manifest["raw_payloads_on_disk"] == "mixed"
    assert len(list(tmp_path.glob("answers/*/*.raw.json"))) == 3


def test_an_orphan_raw_file_from_a_crash_is_ignored_then_overwritten(tmp_path):
    config = make_config()
    orphan = store.raw_path(tmp_path, "anthropic", 0, 0)
    store.write_raw_payload(orphan, {"stale": True})
    assert store.load_answers(tmp_path) == []
    _start(tmp_path, config, {"anthropic": _priced_engine(raw_payload={"fresh": True})})
    assert store.read_json(orphan) == {"fresh": True}
    record = store.read_json(store.answer_path(tmp_path, "anthropic", 0, 0))
    assert record["raw_sha256"] == store.write_raw_payload(orphan, {"fresh": True})["sha256"]


def test_raw_file_is_written_before_the_record(tmp_path, monkeypatch):
    config = make_config()
    real_write = store.write_json

    def refuse_answers(path, document):
        if document.get("schema") == store.ANSWER_SCHEMA:
            raise OSError("disk full")
        real_write(path, document)
    monkeypatch.setattr(store, "write_json", refuse_answers)
    try:
        _start(tmp_path, config, {"anthropic": _priced_engine(raw_payload={"k": 1})},
               max_calls=1)
    except OSError:
        pass
    assert store.raw_path(tmp_path, "anthropic", 0, 0).exists()
    assert store.load_answers(tmp_path) == []


def test_cost_basis_names_the_dates_the_answers_were_priced_at():
    from answersnap.metrics.usage import cost_totals
    records = [{"cost_estimate_usd": 0.1, "cost_basis": {"as_of": "2026-10-05"}},
               {"cost_estimate_usd": 0.2, "cost_basis": {"as_of": "2027-01-02"}},
               {"cost_estimate_usd": None, "cost_basis": {"kind": "none"}}]
    totals = cost_totals(records)
    assert totals["basis"] == "list price as of 2026-10-05, 2027-01-02"
    assert totals["answers_unpriced"] == 1
