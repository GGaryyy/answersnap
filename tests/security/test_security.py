"""Defensive checks: answer text and URLs are untrusted, keys never reach disk."""

from pathlib import Path

import pytest
from builders import STAMP, citation, make_config, record

from answersnap import cli, store
from answersnap.report.build import build_report
from answersnap.report.html import render_html
from answersnap.report.markdown import render_markdown

PAYLOAD = '<script>alert("x")</script><img src=x onerror=alert(1)>'


def _manifest():
    engines = {name: {"status": "ok", "reason": None, "models_reported": ["m"], "planned": 1,
                      "done": 1, "failed": 0} for name in ("anthropic", "openai")}
    return {"run_id": "R", "mode": "live", "status": "complete", "started_at": STAMP.isoformat(),
            "finished_at": STAMP.isoformat(), "repeats": 1, "tool_version": "t",
            "prompt_set": {"version": 1, "hash": "h", "n_queries": 2}, "config_hash": "c",
            "calls": {"planned": 1, "made": 1, "failed": 0, "cap": None}, "engines": engines}


def _hostile_report():
    config = make_config(brand={"name": "Trellis", "owned_domains": ["trellis.example"]})
    hostile = record(text=f"Trellis {PAYLOAD} is fine.",
                     citations=[citation(f'https://trellis.example/"><script>1</script>')])
    hostile["query_text"] = PAYLOAD
    return build_report(_manifest(), config, [hostile], [], {}, {"method_version": "m",
                        "passed": True, "controls": []}, generated_at=STAMP)


def test_html_report_escapes_answer_text_queries_and_urls():
    html = render_html(_hostile_report())
    assert "<script>alert" not in html
    assert "<img src=x" not in html
    assert '"><script>1</script>' not in html
    assert "&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt;" in html


def test_markdown_table_cells_cannot_break_out_of_their_row():
    report = _hostile_report()
    report["rows"][0]["query_text"] = "a | b\n| injected | row"
    lines = [line for line in render_markdown(report).splitlines() if "injected" in line]
    assert len(lines) == 1 and "a \\| b" in lines[0]


class _AlternatingClient:
    """Stands in for the Anthropic SDK: every other call fails at transport."""

    def __init__(self, response):
        self.messages = self
        self._response = response
        self.count = 0

    def create(self, **kwargs):
        import anthropic
        self.count += 1
        if self.count % 2 == 0:
            # No request object: the SDK's HTTP client is an implementation
            # detail this package does not depend on.
            raise anthropic.APIConnectionError(request=None)
        return self._response


def test_api_keys_never_reach_the_run_directory_on_a_live_run(tmp_path, monkeypatch):
    from fake_answers import FakeAnthropicResponse, anthropic_citation, anthropic_text_block

    from answersnap.providers.anthropic_provider import AnthropicProvider

    secret = "sk-test-SECRET-should-never-be-written"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    response = FakeAnthropicResponse([
        {"type": "web_search_tool_result", "caller": {"type": "direct"}, "content": []},
        anthropic_text_block("Example Coffee Co. is good.",
                             [anthropic_citation("https://example-coffee.example/x", cited_text="q")])])
    client = _AlternatingClient(response)
    monkeypatch.setattr(AnthropicProvider, "_client_or_fail", lambda self: client)

    config_path = tmp_path / "c.yaml"
    cli.main(["init", "--out", str(config_path)])
    cli.main(["run", "--config", str(config_path), "--out", str(tmp_path / "out"),
              "--engines", "anthropic", "--yes", "--no-fetch", "--workers", "1"])
    written = [p for p in (tmp_path / "out").rglob("*") if p.is_file()]
    assert any(p.name.endswith(".error.json") for p in written)      # error path exercised
    assert any(p.name.endswith("_r0.json") for p in written)         # success path exercised
    assert client.count == 12
    assert not any(secret in p.read_text(encoding="utf-8", errors="ignore") for p in written)


def test_engine_names_in_attributes_are_escaped_too():
    report = _hostile_report()
    report["engines"][0]["engine"] = '"><script>alert(2)</script>'
    report["headline"]["per_engine"][0]["engine"] = '"><script>alert(2)</script>'
    html = render_html(report)
    assert "<script>alert(2)" not in html


def test_markdown_cannot_carry_html_or_links_from_answers():
    report = _hostile_report()
    report["rows"][0]["brand_segments"] = [("[click](javascript:alert(1)) <b>x</b>", False)]
    # URLs are rendered as code spans, whose content Markdown shows literally;
    # backticks are stripped from them, so nothing can close the span early.
    import re
    outside_code = re.sub(r"`[^`]*`", "", render_markdown(report))
    assert "<script>" not in outside_code and "<img" not in outside_code
    assert "<b>" not in outside_code
    # Both brackets escaped, so no Markdown link can form.
    assert "\\[click\\](javascript:" in outside_code
    assert not re.search(r"(?<!\\)\[click(?<!\\)\]\(", outside_code)


@pytest.mark.parametrize("name", ["../../etc", "..", "/abs/path", "a/../../b"])
def test_brand_names_cannot_escape_the_output_root(tmp_path, name):
    target = store.brand_dir(tmp_path, name).resolve()
    assert target.parent == tmp_path.resolve()


def test_cache_paths_are_hashes_not_urls(tmp_path):
    path = store.fetched_path(tmp_path, "https://evil.example/../../../etc/passwd")
    assert path.parent == tmp_path / "fetched"
    assert ".." not in path.name and "/" not in path.name


def test_package_holds_no_credentials():
    root = Path(__file__).resolve().parents[2] / "src" / "answersnap"
    patterns = ("sk-ant-", "sk-proj-", "AIza", "-----BEGIN")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".j2", ".html"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not any(p in text for p in patterns), path


def _hostile_record_2_report():
    config = make_config(brand={"name": "Trellis", "owned_domains": ["trellis.example"]})
    text = f"Trellis {PAYLOAD} is fine."
    hostile = record(text=text, citations=[citation("https://trellis.example/a")],
                     searches=[{"query": PAYLOAD, "results_count": 1}], search_count=1,
                     spans=[{"start": 0, "end": len(text), "citation_indexes": [0]}])
    return build_report(_manifest(), config, [hostile], [], {}, {"method_version": "m",
                        "passed": True, "controls": []}, generated_at=STAMP)


def test_search_queries_and_span_excerpts_are_escaped_in_html():
    report = _hostile_record_2_report()
    assert report["rows"][0]["owned_cited_sources"][0]["excerpts"]
    html = render_html(report)
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "<mark>Trellis &lt;script&gt;" in html


def test_search_queries_and_span_excerpts_are_inert_in_markdown():
    markdown = render_markdown(_hostile_record_2_report())
    assert "<script>" not in markdown and "<img" not in markdown
    assert "&lt;script&gt;" in markdown
