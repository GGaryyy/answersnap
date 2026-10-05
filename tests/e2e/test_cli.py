"""The commands a user types, end to end."""

import json
import socket

import pytest
import yaml

from answersnap import cli, store


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access during a dry run")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "answersnap.yaml"
    assert cli.main(["init", "--out", str(path)]) == cli.EXIT_OK
    return path


def _run_dir(out):
    (brand,) = out.iterdir()
    (run_dir,) = brand.iterdir()
    return run_dir


def test_init_writes_the_example_and_refuses_to_overwrite(config_path, capsys):
    assert yaml.safe_load(config_path.read_text())["brand"]["name"] == "Example Coffee Co."
    assert cli.main(["init", "--out", str(config_path)]) == cli.EXIT_USAGE
    assert "not overwriting" in capsys.readouterr().out


def test_dry_run_produces_every_output_with_zero_network(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    assert cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out)]) == cli.EXIT_OK
    run_dir = _run_dir(out)
    manifest = store.read_manifest(run_dir)
    assert manifest["mode"] == "dry_run" and manifest["status"] == "complete"
    assert manifest["calls"] == {"planned": 36, "made": 36, "failed": 0, "cap": None,
                                "this_session": 36}
    for engine in ("anthropic", "openai", "google"):
        assert len(list((run_dir / "answers" / engine).glob("q*_r*.json"))) == 12
    for name in ("report.json", "report.md", "report.html", "config.frozen.json"):
        assert (run_dir / name).is_file()
    assert "DRY RUN" in (run_dir / "report.html").read_text(encoding="utf-8")


def test_dry_run_of_your_own_questions_shows_placeholders_not_numbers(tmp_path, no_network):
    document = yaml.safe_load(cli.resources.files("answersnap.examples")
                              .joinpath("example.yaml").read_text())
    document["prompt_set"]["queries"] = [{"text": "Something only I would ask?",
                                          "intent": "recommendation"}]
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump(document))
    out = tmp_path / "out"
    cli.main(["run", "--config", str(path), "--dry-run", "--out", str(out)])
    data = json.loads((_run_dir(out) / "report.json").read_text())
    assert data["rows"] == []
    assert all(e["metrics"] is None for e in data["engines"])


def test_report_rebuilds_from_the_directory_alone(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out)])
    run_dir = _run_dir(out)
    config_path.write_text(config_path.read_text().replace("Example Coffee Co.", "Renamed"))
    (run_dir / "report.md").unlink()
    assert cli.main(["report", str(run_dir)]) == cli.EXIT_OK
    assert "Example Coffee Co." in (run_dir / "report.md").read_text(encoding="utf-8")


def test_live_run_without_keys_spends_nothing_and_says_why(config_path, tmp_path, monkeypatch, capsys):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "out"
    code = cli.main(["run", "--config", str(config_path), "--out", str(out), "--yes", "--no-fetch"])
    assert code == cli.EXIT_INCOMPLETE
    text = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY is not set" in text and "Plan: 0 calls" in text
    manifest = store.read_manifest(_run_dir(out))
    assert {e["status"] for e in manifest["engines"].values()} == {"skipped_no_credentials"}


def test_live_run_off_a_terminal_needs_yes(config_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    out = tmp_path / "out"
    code = cli.main(["run", "--config", str(config_path), "--out", str(out), "--engines", "anthropic"])
    assert code == cli.EXIT_USAGE
    assert "confirm with --yes" in capsys.readouterr().out
    assert not out.exists()


def test_edited_questions_at_the_same_version_are_refused(config_path, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out)])
    run_dir = _run_dir(out)
    manifest = store.read_manifest(run_dir)
    manifest["mode"] = "live"   # pretend the earlier run was real
    store.write_manifest(run_dir, {k: v for k, v in manifest.items() if k != "schema"})
    config_path.write_text(config_path.read_text().replace("stay fresh?", "stay fresh for?"))
    code = cli.main(["run", "--config", str(config_path), "--out", str(out), "--yes"])
    assert code == cli.EXIT_USAGE
    assert "Bump prompt_set.version" in capsys.readouterr().out


def test_bad_config_is_a_usage_error_with_its_path(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("schema: 1\nbrand: {name: X, alias: [y]}\nprompt_set: {version: 1, queries: []}\n")
    assert cli.main(["run", "--config", str(path), "--dry-run"]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert "brand.alias" in out and "prompt_set.queries" in out


def test_engines_flag_must_name_configured_engines(config_path, capsys):
    assert cli.main(["run", "--config", str(config_path), "--dry-run",
                     "--engines", "perplexity"]) == cli.EXIT_USAGE
    assert "not in the config's engines" in capsys.readouterr().out


def test_max_calls_marks_the_run_incomplete(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    code = cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out),
                     "--max-calls", "5"])
    assert code == cli.EXIT_INCOMPLETE
    assert store.read_manifest(_run_dir(out))["calls"]["made"] == 5


def test_resume_fills_in_a_capped_dry_run(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out), "--max-calls", "5"])
    run_dir = _run_dir(out)
    assert cli.main(["run", "--resume", str(run_dir), "--dry-run"]) == cli.EXIT_OK
    assert store.read_manifest(run_dir)["status"] == "complete"
    assert len(store.done_pairs(run_dir)) == 36


def test_verify_providers_lists_supported_and_verified(capsys):
    assert cli.main(["verify-providers"]) == cli.EXIT_OK
    lines = {line.split()[0]: line.split()[1:] for line in capsys.readouterr().out.splitlines()[1:]}
    assert lines["anthropic"] == ["yes", "yes"]
    assert lines["perplexity"] == ["no", "no"]


def test_doctor_reports_keys_and_config(config_path, monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert cli.main(["doctor", "--config", str(config_path)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY or GOOGLE_API_KEY not set" in out and "config ok: Example Coffee Co." in out


def test_doctor_without_any_ready_engine_is_not_healthy(monkeypatch, capsys):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert cli.main(["doctor"]) == cli.EXIT_INCOMPLETE
    assert "no engine is ready" in capsys.readouterr().out


def test_no_command_prints_help():
    assert cli.main([]) == cli.EXIT_USAGE


# ---------------------------------------------------------------- review regressions
def test_resuming_a_subset_never_hides_the_other_engines(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out), "--max-calls", "4"])
    run_dir = _run_dir(out)
    code = cli.main(["run", "--resume", str(run_dir), "--dry-run", "--engines", "anthropic"])
    manifest = store.read_manifest(run_dir)
    assert code == cli.EXIT_INCOMPLETE
    assert manifest["status"] == "incomplete"
    assert manifest["engines"]["anthropic"]["status"] == "ok"
    for engine in ("openai", "google"):
        entry = manifest["engines"][engine]
        assert (entry["status"], entry["done"], entry["planned"]) == ("partial", 1, 12)


@pytest.mark.parametrize("first, resume_flags", [
    (["--dry-run"], []),            # dry run resumed live
    ([], ["--dry-run"]),            # live run resumed from fixtures
])
def test_resume_refuses_to_mix_live_and_fixture_answers(config_path, tmp_path, monkeypatch,
                                                        capsys, first, resume_flags):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--out", str(out), "--yes", "--no-fetch", *first])
    run_dir = _run_dir(out)
    before = sorted(p.name for p in run_dir.rglob("*.json"))
    assert cli.main(["run", "--resume", str(run_dir), "--yes", *resume_flags]) == cli.EXIT_USAGE
    assert "resume it" in capsys.readouterr().out
    assert sorted(p.name for p in run_dir.rglob("*.json")) == before


def test_resume_keeps_the_repeats_it_started_with(config_path, tmp_path, no_network, capsys):
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out), "--max-calls", "4"])
    assert cli.main(["run", "--resume", str(_run_dir(out)), "--dry-run", "--repeats", "1"]) == cli.EXIT_USAGE
    assert "cannot be combined with --resume" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "51", "many"])
def test_repeats_outside_the_allowed_range_are_rejected(config_path, value):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--config", str(config_path), "--dry-run", "--repeats", value])
    assert exit_info.value.code == 2


def test_calls_are_counted_across_sessions(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out), "--max-calls", "10"])
    run_dir = _run_dir(out)
    cli.main(["run", "--resume", str(run_dir), "--dry-run"])
    calls = store.read_manifest(run_dir)["calls"]
    assert (calls["planned"], calls["made"], calls["this_session"]) == (36, 36, 26)


def test_subset_run_says_which_engines_were_not_asked(config_path, tmp_path, no_network):
    out = tmp_path / "out"
    assert cli.main(["run", "--config", str(config_path), "--dry-run", "--out", str(out),
                     "--engines", "anthropic"]) == cli.EXIT_OK
    data = json.loads((_run_dir(out) / "report.json").read_text())
    assert data["run"]["status"] == "complete_subset"
    assert any(n.startswith("Not asked in this run: ChatGPT, Gemini") for n in data["notices"])
