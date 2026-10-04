"""Turn a run directory into report.json, report.md and report.html."""

from datetime import datetime, timezone
from pathlib import Path

from answersnap import faithfulness, run, store
from answersnap.config import parse_config
from answersnap.report.build import build_report
from answersnap.report.html import render_html
from answersnap.report.markdown import render_markdown

# ---------------------------------------------------------------- constants
REPORT_FILES = ("report.json", "report.md", "report.html")


def choose_fetcher(mode, fetch_enabled):
    if not fetch_enabled:
        return None
    return faithfulness.FixtureFetcher() if mode == "dry_run" else faithfulness.LiveFetcher()


def generate(run_dir, *, fetch=True, clock=None):
    """Rebuild every report file from the run directory alone. Returns paths."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    run_dir = Path(run_dir)
    manifest = store.read_manifest(run_dir)
    # A run killed hard never reached finalise; its manifest still says
    # "running" with whatever the session had counted. Settle it from disk so
    # the report never presents those counts as the run's.
    if manifest["status"] == run.STATUS_RUNNING:
        run.settle(manifest, run_dir)
    config = parse_config(store.read_frozen_config(run_dir), source=str(run_dir))
    records = store.load_answers(run_dir)
    fetcher = choose_fetcher(manifest["mode"], fetch)

    faith_rows = faithfulness.check(run_dir, records, fetcher=fetcher, clock=clock)
    faith_summary = faithfulness.summarise(faith_rows, config.engines, fetched=fetcher is not None)
    manifest["faithfulness"] = {"status": "checked" if fetcher else "not_checked",
                                "method_version": faithfulness.METHOD_VERSION,
                                "rows": len(faith_rows),
                                "pages_cached": len(list((run_dir / store.FETCHED_DIR).glob("*.json")))}
    store.write_manifest(run_dir, {k: v for k, v in manifest.items() if k != "schema"})

    report = build_report(manifest, config, records, faith_rows, faith_summary,
                          faithfulness.instrument_check(), generated_at=clock(),
                          errors=store.load_errors(run_dir))
    paths = [run_dir / name for name in REPORT_FILES]
    store.write_json(paths[0], report)
    paths[1].write_text(render_markdown(report), encoding="utf-8")
    paths[2].write_text(render_html(report), encoding="utf-8")
    return paths
