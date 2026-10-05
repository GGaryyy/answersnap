"""Snapshots on disk: one JSON file per answer, plus a manifest per run.

Plain files rather than a database so that anyone can audit a run with a text
editor and `git diff`: the report is computed from these files and nothing else.

    <out>/<brand-slug>/<run_id>/
        manifest.json          what was asked, of whom, when, and how far it got
        config.frozen.json     the config as it was at run time
        answers/<engine>/q00_r0.json         one answer, raw text and citations
        answers/<engine>/q00_r0.raw.json     the API response as received (--no-raw skips)
        answers/<engine>/q00_r0.error.json   a call that failed, kept, not hidden
        fetched/<hash>.json    a cited page as fetched, for the faithfulness check
        report.json / report.md / report.html
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from answersnap.config import brand_slug

# ---------------------------------------------------------------- constants
ANSWER_SCHEMA = "answer-2"
RAW_SUFFIX = ".raw.json"
# Only these are answers; .error.json, .raw.json and .tmp files sit beside them.
ANSWER_FILE = re.compile(r"^q\d{2,}_r\d+\.json$")
RAW_KEPT, RAW_NOT_KEPT, RAW_NOT_PROVIDED = "kept", "not_kept", "not_provided"
# What an answer-1 record (or any record written before a field existed) reads
# as: unknown, never zero.
ANSWER_2_DEFAULTS = {
    "searches": None, "search_count": None, "spans": None, "spans_dropped": None,
    "usage": None, "raw_status": None, "raw_file": None, "raw_sha256": None,
    "raw_bytes": None, "cost_estimate_usd": None, "cost_basis": None,
}
ERROR_SCHEMA = "error-1"
MANIFEST_SCHEMA = "manifest-1"
MANIFEST_NAME = "manifest.json"
FROZEN_CONFIG_NAME = "config.frozen.json"
ANSWERS_DIR = "answers"
FETCHED_DIR = "fetched"
RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"


class PromptSetChanged(RuntimeError):
    """An earlier run answered different questions under the same version."""


def new_run_id(now=None):
    return (now or datetime.now(timezone.utc)).strftime(RUN_ID_FORMAT)


def brand_dir(out_root, brand_name):
    return Path(out_root) / brand_slug(brand_name)


def new_run_dir(out_root, brand_name, now=None):
    """(run_dir, run_id) for a fresh run. Picks the name only; creates nothing,
    so a run refused at the confirmation prompt leaves no empty directory.
    Two runs started within the same second get distinct directories."""
    base = new_run_id(now)
    root = brand_dir(out_root, brand_name)
    run_id, suffix = base, 2
    while (root / run_id).exists():
        run_id, suffix = f"{base}-{suffix}", suffix + 1
    return root / run_id, run_id


def write_json(path, document):
    """Write via a temporary file, so an interrupted run never leaves half a
    JSON file behind — resume would otherwise count it as done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_raw_payload(path, payload):
    """Write the response as received; return its size and digest for the record."""
    data = (json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_atomic(path, data)
    return {"sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def answer_path(run_dir, engine, query_index, repeat):
    return Path(run_dir) / ANSWERS_DIR / engine / f"q{query_index:02d}_r{repeat}.json"


def error_path(run_dir, engine, query_index, repeat):
    return Path(run_dir) / ANSWERS_DIR / engine / f"q{query_index:02d}_r{repeat}.error.json"


def raw_path(run_dir, engine, query_index, repeat):
    return Path(run_dir) / ANSWERS_DIR / engine / f"q{query_index:02d}_r{repeat}{RAW_SUFFIX}"


def fetched_path(run_dir, url):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return Path(run_dir) / FETCHED_DIR / f"{digest}.json"


def _iso(moment):
    return moment.isoformat() if moment else None


def _citation_record(citation):
    return {"url": citation.url, "title": citation.title, "domain": citation.domain,
            "is_cited": citation.is_cited, "position": citation.position,
            "cited_text": citation.cited_text}


def _search_record(search):
    return {"query": search.query, "results_count": search.results_count}


def _span_record(span):
    return {"start": span.start, "end": span.end,
            "citation_indexes": list(span.citation_indexes)}


def _optional(items, convert):
    return None if items is None else [convert(item) for item in items]


def raw_fields(status, file=None, sha256=None, size=None):
    return {"raw_status": status, "raw_file": file, "raw_sha256": sha256, "raw_bytes": size}


def answer_record(answer, *, run_id, query_index, query, repeat, prompt_set_version,
                  prompt_set_hash, raw=None, cost=None):
    """Everything the report needs, and nothing normalised: the answer text is
    stored exactly as the engine returned it, because every evidence offset in
    the report points into it."""
    params = dict(answer.params or {})
    return {
        "schema": ANSWER_SCHEMA,
        "run_id": run_id,
        "engine": answer.platform,
        "model": answer.model,
        "model_requested": params.get("model_requested"),
        "model_reported_by_platform": params.get("model_reported_by_platform"),
        "params": params,
        "query_index": query_index,
        "query_text": query.text,
        "intent": query.intent,
        "repeat": repeat,
        "requested_at": _iso(answer.requested_at),
        "observed_at": _iso(answer.observed_at),
        "answer_text": answer.answer_text,
        "citations": [_citation_record(c) for c in answer.citations],
        # Whether THIS call could show citations at all. False means "we could
        # not see", which must never be read as "nothing was cited".
        "cited_sources_available": answer.cited_sources_available,
        "retrieved_set_available": answer.retrieved_set_available,
        "stop_reason": answer.stop_reason,
        "prompt_set_version": prompt_set_version,
        "prompt_set_hash": prompt_set_hash,
        # None in any of these means the platform did not say, never "none".
        "searches": _optional(answer.searches, _search_record),
        "search_count": answer.search_count,
        "spans": _optional(answer.spans, _span_record),
        "spans_dropped": answer.spans_dropped,
        "usage": answer.usage,
        **(raw or raw_fields(RAW_NOT_PROVIDED if answer.raw_payload is None else RAW_NOT_KEPT)),
        "cost_estimate_usd": cost[0] if cost else None,
        "cost_basis": cost[1] if cost else None,
    }


def error_record(*, run_id, engine, query_index, repeat, requested_at, error):
    # The adapter's message names the failure type only; it never carries the
    # request body or a key, so it is safe to keep on disk.
    return {"schema": ERROR_SCHEMA, "run_id": run_id, "engine": engine,
            "query_index": query_index, "repeat": repeat,
            "requested_at": _iso(requested_at),
            "error_type": type(error).__name__, "message": str(error)}


def _answer_files(run_dir):
    root = Path(run_dir) / ANSWERS_DIR
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*/*.json") if ANSWER_FILE.match(p.name))


def load_answers(run_dir):
    records = []
    for path in _answer_files(run_dir):
        record = read_json(path)
        for key, value in ANSWER_2_DEFAULTS.items():
            record.setdefault(key, value)
        record["_file"] = path.relative_to(run_dir).as_posix()
        records.append(record)
    return records


def load_errors(run_dir):
    root = Path(run_dir) / ANSWERS_DIR
    if not root.is_dir():
        return []
    return [read_json(p) for p in sorted(root.glob("*/*.error.json"))]


def done_pairs(run_dir):
    """(engine, query_index, repeat) already answered. Failed calls are not
    done: resume retries them."""
    return {(r["engine"], r["query_index"], r["repeat"]) for r in load_answers(run_dir)}


def write_manifest(run_dir, manifest):
    write_json(Path(run_dir) / MANIFEST_NAME, {"schema": MANIFEST_SCHEMA, **manifest})


def read_manifest(run_dir):
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} is not an answersnap run (no {MANIFEST_NAME})")
    return read_json(path)


def write_frozen_config(run_dir, config_document):
    write_json(Path(run_dir) / FROZEN_CONFIG_NAME, config_document)


def read_frozen_config(run_dir):
    return read_json(Path(run_dir) / FROZEN_CONFIG_NAME)


def _earlier_live_manifests(brand_root):
    if not brand_root.is_dir():
        return []
    manifests = []
    for path in sorted(brand_root.glob(f"*/{MANIFEST_NAME}")):
        manifest = read_json(path)
        # Dry runs answer from fixtures; editing questions after one is normal.
        if manifest.get("mode") == "live":
            manifests.append((path.parent, manifest))
    return manifests


def refuse_if_prompt_set_changed(out_root, brand_name, version, prompt_hash):
    """Refuse a run whose questions differ from an earlier run's at the same
    version. Rates are only comparable across identical questions, so a silent
    in-place edit would break every comparison drawn across it."""
    for run_dir, manifest in _earlier_live_manifests(brand_dir(out_root, brand_name)):
        earlier = manifest.get("prompt_set") or {}
        if earlier.get("version") == version and earlier.get("hash") != prompt_hash:
            raise PromptSetChanged(
                f"prompt_set version {version} was already answered with different "
                f"questions in {run_dir}. Bump prompt_set.version for the edited "
                "questions; answers to different questions must not share a version.")
