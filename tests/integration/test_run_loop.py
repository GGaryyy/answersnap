"""The sampling loop with stand-in engines: what gets asked, written and counted."""

from builders import STAMP, make_config

from answersnap import run, store
from answersnap.providers import ProviderAnswer, ProviderError


class ScriptedEngine:
    """Answers from a script; raises ProviderError for listed (query, repeat)."""

    mode = "live"

    def __init__(self, name, fail=(), model="model-x"):
        self.name = name
        self.model = model
        self.asked = []
        self._fail = set(fail)

    def answer(self, query_text, repeat):
        self.asked.append((query_text, repeat))
        if (query_text, repeat) in self._fail:
            raise ProviderError(f"{self.name} request failed: RateLimitError")
        return ProviderAnswer(platform=self.name, model=self.model,
                              params={"model_requested": self.model}, requested_at=STAMP,
                              observed_at=STAMP, answer_text=f"Trellis says {self.name}",
                              cited_sources_available=True)


def _start(tmp_path, config, engines, skipped=None, max_calls=None, done=frozenset()):
    tasks = run.plan_tasks(config, list(engines), config.repeats, done)
    if max_calls is not None:
        tasks = tasks[:max_calls]
    manifest = run.initial_manifest(config, run_id="R1", mode="live", engines=engines,
                                    skipped=skipped or {}, tasks=tasks, repeats=config.repeats,
                                    max_calls=max_calls, started_at=STAMP,
                                    cost=run.estimate_cost(tasks))
    run.execute(config, tmp_path, manifest, engines, tasks, workers=1, clock=lambda: STAMP)
    return run.finalise(manifest, tmp_path, STAMP), tasks


def test_plan_is_repeat_major_and_skips_done_pairs():
    config = make_config()
    tasks = run.plan_tasks(config, ["anthropic", "openai"], 2, done={("openai", 0, 0)})
    assert [(t.engine, t.query_index, t.repeat) for t in tasks[:3]] == [
        ("anthropic", 0, 0), ("anthropic", 1, 0), ("openai", 1, 0)]
    assert len(tasks) == 2 * 2 * 2 - 1


def test_estimate_has_a_figure_only_where_cost_was_measured():
    config = make_config()
    estimate = run.estimate_cost(run.plan_tasks(config, ["anthropic", "openai"], 2))
    assert estimate["anthropic"] == (4, 0.68)
    assert estimate["openai"] == (4, None)


def test_every_question_is_sent_verbatim_and_written(tmp_path):
    config = make_config()
    engines = {"anthropic": ScriptedEngine("anthropic"), "openai": ScriptedEngine("openai")}
    manifest, _ = _start(tmp_path, config, engines)
    texts = [q.text for q in config.prompt_set.queries]
    assert sorted(engines["anthropic"].asked) == sorted((t, r) for t in texts for r in range(2))
    assert len(store.load_answers(tmp_path)) == 8
    assert manifest["status"] == run.STATUS_COMPLETE
    assert manifest["engines"]["anthropic"] == {
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
