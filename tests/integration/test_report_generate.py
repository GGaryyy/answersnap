"""A full dry-run directory turned into reports, checked for what auditors rely on."""

import json
import re
from html.parser import HTMLParser
from importlib import resources

import pytest
from builders import STAMP

from answersnap import report, run, store
from answersnap.config import load_config
from answersnap.engines import build_engines

EXAMPLE = resources.files("answersnap.examples").joinpath("example.yaml")


@pytest.fixture
def dry_run_dir(tmp_path):
    config = load_config(EXAMPLE)
    engines, _ = build_engines(config.engines, dry_run=True, clock=lambda: STAMP)
    tasks = run.plan_tasks(config, list(engines), config.repeats)
    manifest = run.initial_manifest(config, run_id="20261001T120000Z", mode="dry_run",
                                    engines=engines, skipped={}, tasks=tasks,
                                    repeats=config.repeats, max_calls=None, started_at=STAMP,
                                    cost=run.estimate_cost(tasks))
    store.write_frozen_config(tmp_path, config.to_dict())
    run.execute(config, tmp_path, manifest, engines, tasks, workers=1, clock=lambda: STAMP)
    run.finalise(manifest, tmp_path, STAMP)
    store.write_manifest(tmp_path, manifest)
    report.generate(tmp_path, clock=lambda: STAMP)
    return tmp_path


def _report(run_dir):
    return json.loads((run_dir / "report.json").read_text(encoding="utf-8"))


def test_headline_is_per_engine_with_a_count_across(dry_run_dir):
    headline = _report(dry_run_dir)["headline"]
    named = {h["engine"]: (h["named_in_rec_answers"]["count"], h["named_in_rec_answers"]["n"])
             for h in headline["per_engine"]}
    assert named == {"anthropic": (4, 6), "openai": (2, 6), "google": (0, 6)}
    assert headline["across_engines"]["count"] == 6 and headline["across_engines"]["n"] == 18


def test_faithfulness_demo_shows_found_partial_and_missing(dry_run_dir):
    data = _report(dry_run_dir)
    anthropic = next(e for e in data["engines"] if e["engine"] == "anthropic")
    assert anthropic["faithfulness"]["status"] == "ok"
    rows = [f for r in data["rows"] if r["engine"] == "anthropic" for f in r["faithfulness"]]
    scores = {f["result"] for f in rows}
    assert {"found", "not_found", "not_checkable"} <= scores
    partial = [f["best_window_score"] for f in rows if f["result"] == "not_found"]
    assert any(0.5 < s < 1 for s in partial) and any(s < 0.5 for s in partial)
    for engine in ("openai", "google"):
        section = next(e for e in data["engines"] if e["engine"] == engine)
        assert section["faithfulness"]["status"] == "not_observable"


def test_report_records_what_makes_it_reproducible(dry_run_dir):
    data = _report(dry_run_dir)
    assert data["run"]["prompt_set"]["hash"].startswith("sha256:")
    assert data["run"]["repeats"] == 3 and data["run"]["mode"] == "dry_run"
    assert {m for e in data["engines"] for m in e["models"]} == {
        "claude-sonnet-5", "gpt-5.1", "gemini-3.7-flash"}
    assert data["faithfulness_instrument"]["passed"] is True
    assert data["notices"][0].startswith("DRY RUN")


def test_every_output_carries_the_definitions(dry_run_dir):
    for name in ("report.md", "report.html"):
        text = (dry_run_dir / name).read_text(encoding="utf-8")
        assert "Named in recommendation answers" in text
        assert "This is not the same as being recommended" in text
    assert len(_report(dry_run_dir)["definitions"]) == 6


def test_the_report_never_calls_its_own_metric_recommended():
    # Answer text may say "recommended" — engines write what they like. The
    # report's own wording may use it only in the disclaimer.
    package = resources.files("answersnap")
    sources = [package.joinpath("report/templates/report.html.j2"),
               package.joinpath("report/markdown.py"), package.joinpath("report/format.py")]
    for source in sources:
        assert not re.search(r"\brecommended\b", source.read_text(encoding="utf-8")), source.name
    from answersnap.report.build import DEFINITIONS
    assert sum(len(re.findall(r"\brecommended\b", text)) for _, text in DEFINITIONS) == 1


class _Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.hrefs = set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.hrefs.append(attrs["href"][1:])


def test_every_headline_link_lands_on_evidence(dry_run_dir):
    parser = _Anchors()
    parser.feed((dry_run_dir / "report.html").read_text(encoding="utf-8"))
    assert parser.hrefs and set(parser.hrefs) <= parser.ids
    assert {"evidence-anthropic", "evidence-openai", "evidence-google"} <= parser.ids
    assert "row-anthropic-q00-r0" in parser.ids


def test_report_is_a_pure_function_of_the_run_directory(dry_run_dir):
    first = (dry_run_dir / "report.json").read_text(encoding="utf-8")
    report.generate(dry_run_dir, clock=lambda: STAMP)
    assert (dry_run_dir / "report.json").read_text(encoding="utf-8") == first


def test_not_recommended_on_a_recommendation_question_is_an_empty_cell(dry_run_dir):
    html = (dry_run_dir / "report.html").read_text(encoding="utf-8")
    row = re.search(r'<tr id="row-google-q00-r0">(.*?)</tr>', html, re.S).group(1)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    assert cells[4].strip() == "" and cells[5].strip() == ""   # mentioned, named
    off = re.search(r'<tr id="row-google-q03-r0">(.*?)</tr>', html, re.S).group(1)
    assert "not a recommendation question" in off


def test_no_fetch_marks_faithfulness_not_checked(dry_run_dir):
    for path in (dry_run_dir / "fetched").glob("*.json"):
        path.unlink()
    report.generate(dry_run_dir, fetch=False, clock=lambda: STAMP)
    data = _report(dry_run_dir)
    anthropic = next(e for e in data["engines"] if e["engine"] == "anthropic")
    assert anthropic["faithfulness"]["status"] == "not_checked"
