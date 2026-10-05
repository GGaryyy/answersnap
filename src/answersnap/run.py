"""Ask every question, on every engine, the configured number of times.

The loop's job is bookkeeping, not judgment: each answer is written as it
arrives, each failure is written as a failure, and the manifest always says how
far the run got — so an interrupted or capped run is visibly partial rather
than quietly smaller.

Every count in a finished manifest is taken from the files on disk, never from
one session's tallies: after a resume, or when only some engines were asked
this time, the session saw only part of the run.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from answersnap import __version__, pricing, store
from answersnap.config import config_hash, prompt_identity
from answersnap.engines import SKIPPED_NO_CREDENTIALS, SKIPPED_UNVERIFIED, per_call_estimate_usd
from answersnap.metrics.rates import is_placeholder
from answersnap.metrics.usage import cost_totals, usage_totals
from answersnap.providers import ProviderError

# ---------------------------------------------------------------- constants
DEFAULT_WORKERS = 3   # concurrent calls; modest, to stay under rate limits
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
# Every engine that was asked finished, but the run deliberately asked only a
# subset of the configured engines (--engines). Not "complete": the report
# would otherwise imply the missing engines were measured.
STATUS_COMPLETE_SUBSET = "complete_subset"
STATUS_INCOMPLETE = "incomplete"
ENGINE_OK = "ok"
ENGINE_PARTIAL = "partial"
SKIPPED_NOT_REQUESTED = "skipped_not_requested"
SKIP_STATUSES = (SKIPPED_NO_CREDENTIALS, SKIPPED_UNVERIFIED, SKIPPED_NOT_REQUESTED)
# What an adapter's parser raises when a platform changes its response shape.
# Recorded as a failed call like any other, so one odd response cannot sink a
# paid run or escape as a traceback with no report written.
PARSE_ERRORS = (KeyError, IndexError, TypeError, ValueError, AttributeError)


@dataclass(frozen=True)
class Task:
    engine: str
    query_index: int
    repeat: int


def plan_tasks(config, engine_names, repeats, done=frozenset()):
    """Every (engine, query, repeat) still to ask, repeat-major.

    Repeat-major so a capped run still covers every question on every engine
    once before it starts on second answers.
    """
    return [Task(engine, query_index, repeat)
            for repeat in range(repeats)
            for query_index in range(len(config.prompt_set.queries))
            for engine in engine_names
            if (engine, query_index, repeat) not in done]


def estimate_cost(tasks, models=None):
    """{engine: (calls, usd or None)} in the order engines first appear.
    None means the engine's model has no list price."""
    calls = {}
    for task in tasks:
        calls[task.engine] = calls.get(task.engine, 0) + 1
    estimate = {}
    for engine, count in calls.items():
        per_call = per_call_estimate_usd(engine, (models or {}).get(engine))
        estimate[engine] = (count, None if per_call is None else round(count * per_call, 2))
    return estimate


def _now():
    return datetime.now(timezone.utc)


def _engine_entry(name, engines, skipped):
    if name in skipped:
        status, reason = skipped[name]
    elif name not in engines:
        status, reason = SKIPPED_NOT_REQUESTED, None
    else:
        status, reason = STATUS_RUNNING, None
    model = engines[name].model if name in engines else None
    return {"status": status, "reason": reason, "model_requested": model,
            "models_reported": [], "planned": 0, "done": 0, "failed": 0}


def initial_manifest(config, *, run_id, mode, engines, skipped, tasks, repeats,
                     max_calls, started_at, cost, keep_raw=True):
    return {
        "run_id": run_id,
        "tool_version": __version__,
        "mode": mode,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "status": STATUS_RUNNING,
        "brand": config.brand.name,
        "config_hash": config_hash(config),
        "prompt_set": {"version": config.prompt_set.version,
                       "hash": prompt_identity(config),
                       "n_queries": len(config.prompt_set.queries)},
        "repeats": repeats,
        "recommendation_intents": list(config.recommendation_intents),
        "engines": {name: _engine_entry(name, engines, skipped) for name in config.engines},
        "calls": {"planned": 0, "made": 0, "failed": 0, "cap": max_calls,
                  "this_session": len(tasks)},
        "cost_estimate_usd": {name: usd for name, (_, usd) in cost.items()},
        "record_schema": store.ANSWER_SCHEMA,
        "raw_payloads": store.RAW_KEPT if keep_raw else store.RAW_NOT_KEPT,
        "faithfulness": {"status": "not_run"},
    }


class _Recorder:
    """Writes each outcome as it arrives; the manifest counts are settled from
    disk afterwards, so these are progress only."""

    def __init__(self, run_dir, manifest, config):
        self._run_dir = run_dir
        self._manifest = manifest
        self._config = config
        self._version = config.prompt_set.version
        self._hash = prompt_identity(config)
        self._keep_raw = manifest.get("raw_payloads", store.RAW_KEPT) == store.RAW_KEPT
        self._lock = threading.Lock()

    def _raw(self, task, answer):
        """Write the response before the record that points at it, so a record
        never names a file that is not there."""
        path = store.raw_path(self._run_dir, task.engine, task.query_index, task.repeat)
        if answer.raw_payload is None:
            return store.raw_fields(store.RAW_NOT_PROVIDED)
        if not self._keep_raw:
            path.unlink(missing_ok=True)  # an earlier attempt's, now not wanted
            return store.raw_fields(store.RAW_NOT_KEPT)
        written = store.write_raw_payload(path, answer.raw_payload)
        return store.raw_fields(store.RAW_KEPT, path.relative_to(self._run_dir).as_posix(),
                                written["sha256"], written["bytes"])

    def _cost(self, answer):
        if self._manifest.get("mode") == "dry_run":
            return pricing.no_estimate(pricing.REASON_DRY_RUN)
        return pricing.estimate_cost_usd(answer.platform, answer.model, answer.usage,
                                         answer.search_count)

    def answered(self, task, answer):
        query = self._config.prompt_set.queries[task.query_index]
        record = store.answer_record(
            answer, run_id=self._manifest["run_id"], query_index=task.query_index,
            query=query, repeat=task.repeat, prompt_set_version=self._version,
            prompt_set_hash=self._hash, raw=self._raw(task, answer), cost=self._cost(answer))
        store.write_json(store.answer_path(self._run_dir, task.engine, task.query_index,
                                           task.repeat), record)
        error_file = store.error_path(self._run_dir, task.engine, task.query_index, task.repeat)
        error_file.unlink(missing_ok=True)  # a resumed retry that now succeeded
        with self._lock:
            self._manifest["calls"]["made"] += 1

    def failed(self, task, requested_at, error):
        store.write_json(
            store.error_path(self._run_dir, task.engine, task.query_index, task.repeat),
            store.error_record(run_id=self._manifest["run_id"], engine=task.engine,
                               query_index=task.query_index, repeat=task.repeat,
                               requested_at=requested_at, error=error))
        with self._lock:
            self._manifest["calls"]["made"] += 1
            self._manifest["calls"]["failed"] += 1


def _ask(engine, config, task, recorder, clock):
    query = config.prompt_set.queries[task.query_index]
    requested_at = clock()
    try:
        answer = engine.answer(query.text, task.repeat)
    except (ProviderError, *PARSE_ERRORS) as exc:
        recorder.failed(task, requested_at, exc)
        return
    recorder.answered(task, answer)


def _run_parallel(tasks, workers, ask):
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(ask, task) for task in tasks]
        for future in futures:
            future.result()
    except BaseException:
        # Ctrl-C or a bug: stop queued calls instead of letting shutdown wait
        # for every one of them — each would still spend money.
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def execute(config, run_dir, manifest, engines, tasks, *, workers=DEFAULT_WORKERS,
            clock=_now):
    """Run tasks, writing as they finish. The manifest is saved even when the
    run is interrupted, so it never claims more than happened."""
    recorder = _Recorder(run_dir, manifest, config)

    def ask(task):
        _ask(engines[task.engine], config, task, recorder, clock)

    try:
        if workers <= 1:
            for task in tasks:
                ask(task)
        else:
            _run_parallel(tasks, workers, ask)
    finally:
        store.write_manifest(run_dir, manifest)
    return manifest


def _counts_on_disk(records, run_dir):
    answered, failed = {}, {}
    for record in records:
        answered.setdefault(record["engine"], []).append(record["model"])
    for record in store.load_errors(run_dir):
        failed[record["engine"]] = failed.get(record["engine"], 0) + 1
    return answered, failed


def raw_status_on_disk(records):
    """kept / not_kept / not_provided, or "mixed" when a resume changed --no-raw.
    Taken from the records, because the manifest only knows this session's flag."""
    statuses = {r.get("raw_status") for r in records} - {None}
    if not statuses:
        return None
    return statuses.pop() if len(statuses) == 1 else "mixed"


def _settle_usage(entry, records):
    entry["usage"] = usage_totals(records)
    entry["cost_from_usage_usd"] = cost_totals(records)


def _settle_engine(entry, models, failed, per_engine):
    entry["done"] = len(models)
    entry["failed"] = failed
    entry["models_reported"] = sorted(set(models))
    if entry["status"] in SKIP_STATUSES and not models and not failed:
        entry["planned"] = 0
        return
    # An engine with anything on disk is part of the run, whatever this session
    # did with it — skipped now or not, its answers are in the report.
    entry["planned"] = per_engine
    entry["status"] = ENGINE_OK if entry["done"] == per_engine else ENGINE_PARTIAL


def _run_status(engines):
    statuses = [e["status"] for e in engines.values()]
    if any(s in (ENGINE_PARTIAL, SKIPPED_NO_CREDENTIALS, SKIPPED_UNVERIFIED) for s in statuses):
        return STATUS_INCOMPLETE
    if SKIPPED_NOT_REQUESTED in statuses:
        return STATUS_COMPLETE_SUBSET
    return STATUS_COMPLETE


def settle(manifest, run_dir, finished_at=None):
    """Recount every engine and every call from the files on disk."""
    records = store.load_answers(run_dir)
    answered, failed = _counts_on_disk(records, run_dir)
    per_engine = manifest["prompt_set"]["n_queries"] * manifest["repeats"]
    answers = [r for r in records if not is_placeholder(r)]
    for name, entry in manifest["engines"].items():
        _settle_engine(entry, answered.get(name, []), failed.get(name, 0), per_engine)
        # Per engine only: token counts and prices are not comparable across
        # platforms, so no run-wide total is ever written.
        _settle_usage(entry, [r for r in answers if r["engine"] == name])
    manifest["raw_payloads_on_disk"] = raw_status_on_disk(records)
    # A resumed answer-1 run holds both schemas; say so instead of hiding it.
    manifest["record_schemas_on_disk"] = sorted({r.get("schema") for r in records
                                                 if r.get("schema")})
    total_failed = sum(failed.values())
    manifest["calls"].update({
        "planned": sum(e["planned"] for e in manifest["engines"].values()),
        "made": sum(len(m) for m in answered.values()) + total_failed,
        "failed": total_failed})
    manifest["status"] = _run_status(manifest["engines"])
    if finished_at is not None:
        manifest["finished_at"] = finished_at.isoformat()
    return manifest


def finalise(manifest, run_dir, finished_at):
    return settle(manifest, run_dir, finished_at)
