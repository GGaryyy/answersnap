import pytest
from builders import citation, record

from answersnap import faithfulness, store

PAGE = """<html><head><title>T</title><script>var quote = "Hidden in a script.";</script></head>
<body><p>Example Coffee roasts within <b>48 hours</b>.</p><style>.x{}</style></body></html>"""


class StubFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url in self.pages:
            return "ok", 200, self.pages[url]
        return "gone", 404, None


def test_visible_text_drops_scripts_and_decodes_entities():
    text = faithfulness.visible_text(PAGE)
    assert "Example Coffee roasts within 48 hours." in text
    assert "Hidden in a script" not in text
    assert faithfulness.visible_text(None) == ""


def _rows(tmp_path, quotes, pages, observable=True):
    rec = record(citations=[citation("https://a.example/p", cited_text=q) for q in quotes],
                 observable=observable)
    fetcher = StubFetcher(pages)
    return faithfulness.check(tmp_path, [rec], fetcher=fetcher), fetcher


def test_quote_on_the_page_is_found_even_across_tags_and_entities(tmp_path):
    rows, _ = _rows(tmp_path, ["Example Coffee roasts within 48 hours"],
                    {"https://a.example/p": PAGE})
    assert rows[0]["result"] == faithfulness.RESULT_FOUND
    assert rows[0]["method_version"] == faithfulness.METHOD_VERSION


def test_quote_absent_from_the_page_is_not_found_with_its_score(tmp_path):
    rows, _ = _rows(tmp_path, ["Every box includes a printed brewing guide"],
                    {"https://a.example/p": PAGE})
    assert rows[0]["result"] == faithfulness.RESULT_NOT_FOUND
    assert 0 <= rows[0]["best_window_score"] < 1


def test_unfetchable_page_keeps_its_row_as_not_checkable(tmp_path):
    rows, _ = _rows(tmp_path, ["anything"], {})
    assert rows[0]["result"] == faithfulness.RESULT_NOT_CHECKABLE
    assert rows[0]["not_comparable_reason"] == "fetch_gone"
    assert rows[0]["best_window_score"] is None


def test_citation_without_quote_is_never_fetched_or_scored(tmp_path):
    rows, fetcher = _rows(tmp_path, [None], {"https://a.example/p": PAGE})
    assert fetcher.calls == []
    assert rows[0]["not_comparable_reason"] == "no_cited_text"
    assert rows[0]["exact_match"] is None


def test_fetching_switched_off_marks_rows_not_fetched(tmp_path):
    rec = record(citations=[citation("https://a.example/p", cited_text="q")])
    rows = faithfulness.check(tmp_path, [rec], fetcher=None)
    assert rows[0]["not_comparable_reason"] == "not_fetched"


def test_retrieved_only_sources_are_not_checked(tmp_path):
    rec = record(citations=[citation("https://a.example/p", is_cited=False, cited_text="q")])
    assert faithfulness.check(tmp_path, [rec], fetcher=StubFetcher({})) == []


def test_each_url_is_fetched_once_and_cached_on_disk(tmp_path):
    recs = [record(repeat=r, citations=[citation("https://a.example/p", cited_text="48 hours")])
            for r in range(3)]
    fetcher = StubFetcher({"https://a.example/p": PAGE})
    faithfulness.check(tmp_path, recs, fetcher=fetcher)
    assert fetcher.calls == ["https://a.example/p"]
    second = StubFetcher({})
    rows = faithfulness.check(tmp_path, recs, fetcher=second)
    assert second.calls == [] and all(r["result"] == "found" for r in rows)
    cached = store.read_json(store.fetched_path(tmp_path, "https://a.example/p"))
    assert cached["text_method"] == faithfulness.TEXT_METHOD


def test_summary_marks_engines_without_quotes_not_observable():
    rows = [
        {"engine": "anthropic", "cited_text": "q", "result": "found"},
        {"engine": "anthropic", "cited_text": "q", "result": "not_found"},
        {"engine": "anthropic", "cited_text": "q", "result": "not_checkable"},
        {"engine": "openai", "cited_text": None, "result": "not_checkable"},
    ]
    summary = faithfulness.summarise(rows, ["anthropic", "openai", "google"])
    assert summary["anthropic"] == {"status": "ok", "cited": 3, "with_quote": 3, "checked": 2,
                                    "found": 1, "not_found": 1, "not_checkable": 1}
    assert summary["openai"]["status"] == faithfulness.STATUS_NOT_OBSERVABLE
    assert summary["google"]["status"] == faithfulness.STATUS_NO_CITATIONS
    assert faithfulness.summarise(rows, ["anthropic"], fetched=False)["anthropic"]["status"] == "not_checked"


def test_instrument_check_passes_on_known_good_controls():
    check = faithfulness.instrument_check()
    assert check["passed"] is True
    assert {c["label"] for c in check["controls"]} == {"zh", "en"}


def test_fixture_fetcher_serves_bundled_pages_and_nothing_else():
    fetch = faithfulness.FixtureFetcher()
    status, http_status, body = fetch("https://example-coffee.example/shipping")
    assert (status, http_status) == ("ok", 200) and "48 hours" in body
    assert fetch("https://elsewhere.example/") == ("not_in_dry_run", None, None)


def test_live_fetcher_honours_robots_before_fetching(monkeypatch):
    class DenyAll:
        def allows(self, url):
            return False

    def must_not_fetch(url):
        raise AssertionError("fetched despite robots.txt")
    monkeypatch.setattr(faithfulness.safe_fetch, "fetch", must_not_fetch)
    assert faithfulness.LiveFetcher(robots=DenyAll())("https://a.example/") == ("robots_blocked", None, None)


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data"])
def test_live_fetcher_refuses_internal_addresses(url):
    class AllowAll:
        def allows(self, url):
            return True
    status, _, detail = faithfulness.LiveFetcher(robots=AllowAll())(url)
    assert status == "blocked_unsafe"
    assert detail in ("loopback", "link_local")


# ---------------------------------------------------------------- review regressions
class _AllowAll:
    def allows(self, url):
        return True


@pytest.mark.parametrize("url", ["https://example.com:99999/x", "http://[::1/x"])
def test_malformed_citation_urls_cost_one_row_not_the_report(url):
    status, _, _ = faithfulness.LiveFetcher(robots=faithfulness.RobotsCache())(url)
    assert status == "invalid_url"
    status, _, _ = faithfulness.LiveFetcher(robots=_AllowAll())(url)
    assert status == "invalid_url"


def test_transient_fetch_failures_are_retried_next_time(tmp_path):
    rec = record(citations=[citation("https://a.example/p", cited_text="48 hours")])
    flaky = lambda url: ("timeout", None, None)  # noqa: E731
    rows = faithfulness.check(tmp_path, [rec], fetcher=flaky)
    assert rows[0]["not_comparable_reason"] == "fetch_timeout"
    assert not store.fetched_path(tmp_path, "https://a.example/p").exists()
    rows = faithfulness.check(tmp_path, [rec], fetcher=StubFetcher({"https://a.example/p": PAGE}))
    assert rows[0]["result"] == faithfulness.RESULT_FOUND


def test_cached_text_from_another_method_is_fetched_again(tmp_path):
    url = "https://a.example/p"
    store.write_json(store.fetched_path(tmp_path, url),
                     {"url": url, "fetch_status": "ok", "http_status": 200,
                      "fetched_at": "x", "text_method": "html-text-0", "text": "stale"})
    fetcher = StubFetcher({url: PAGE})
    rows = faithfulness.check(tmp_path, [record(citations=[citation(url, cited_text="48 hours")])],
                              fetcher=fetcher)
    assert fetcher.calls == [url] and rows[0]["result"] == faithfulness.RESULT_FOUND
    assert store.read_json(store.fetched_path(tmp_path, url))["text_method"] == faithfulness.TEXT_METHOD


def test_all_quotes_uncheckable_is_a_status_not_zero_found():
    rows = [{"engine": "anthropic", "cited_text": "q", "result": "not_checkable"}] * 5
    summary = faithfulness.summarise(rows, ["anthropic"])["anthropic"]
    assert summary["status"] == faithfulness.STATUS_NOT_CHECKABLE and summary["checked"] == 0



# ---------------------------------------------------------------- live-run regressions
# Found on the first real Claude run: every checkable quote came back "not
# found" even at overlap 1.0. Hand-written fixtures could not show it.

def test_inline_markup_does_not_split_a_sentence():
    page = '<p>brands like <a href="/x">Brand A Co.</a>, which roast for espresso.</p>'
    assert faithfulness.visible_text(page) == "brands like Brand A Co., which roast for espresso."


def test_block_elements_still_separate_words():
    page = "<h2>Best picks</h2><p>Brand D</p><ul><li>Brand C</li><li>Brand E</li></ul>"
    assert faithfulness.visible_text(page) == "Best picks Brand D Brand C Brand E"


@pytest.mark.parametrize("quote, words", [
    ("## Top picks 2026\n### Brand D\n\n\nBrand D connects subscribers...",
     "Top picks 2026 Brand D Brand D connects subscribers"),
    ("**Roast style:** light; single-origin filter focus.", "Roast style: light; single-origin filter focus."),
    ("Winners:\n\n- **Best overall:** Brand B Coffee \u2014 specialty beans",
     "Winners: Best overall: Brand B Coffee \u2014 specialty beans"),
    ("See [the guide](https://a.example/g) for more\u2026", "See the guide for more"),
    ("Plain sentence with 2. numbers inside.", "Plain sentence with 2. numbers inside."),
])
def test_quotes_lose_their_markdown_but_keep_their_words(quote, words):
    assert faithfulness.normalise_quote(quote) == words


def test_a_markdown_quote_matches_the_plain_page_it_came_from(tmp_path):
    page = ("<html><body><h2>Top picks 2026</h2><h3>Brand D</h3>"
            "<p>Brand D connects subscribers with over 50 roasters.</p></body></html>")
    quote = "## Top picks 2026\n### Brand D\n\n\nBrand D connects subscribers..."
    rows, _ = _rows(tmp_path, [quote], {"https://a.example/p": page})
    assert rows[0]["result"] == faithfulness.RESULT_FOUND
    assert rows[0]["cited_text"] == quote   # the reader still sees what the engine said



def test_a_quote_stitched_from_title_and_passage_is_found_piecewise(tmp_path):
    page = ("<html><head><title>Best Roasters 2026</title></head><body><nav>Cart</nav>"
            "<p>Intro text.</p><p>Brand C Lab \u2014 precision-roasted single origins.</p></body></html>")
    quote = "Best Roasters 2026\nBrand C Lab \u2014 precision-roasted single origins."
    rows, _ = _rows(tmp_path, [quote], {"https://a.example/p": page})
    assert rows[0]["result"] == faithfulness.RESULT_FOUND
    assert (rows[0]["matched_by"], rows[0]["pieces"]) == ("pieces", 2)


def test_one_missing_piece_means_not_found(tmp_path):
    page = "<html><body><p>Best Roasters 2026</p><p>Something else entirely.</p></body></html>"
    quote = "Best Roasters 2026\nBrand C Lab \u2014 precision-roasted single origins."
    rows, _ = _rows(tmp_path, [quote], {"https://a.example/p": page})
    assert rows[0]["result"] == faithfulness.RESULT_NOT_FOUND
    assert (rows[0]["pieces"], rows[0]["pieces_found"]) == (2, 1)


def test_a_contiguous_quote_is_marked_as_matched_whole(tmp_path):
    rows, _ = _rows(tmp_path, ["Example Coffee roasts within 48 hours"], {"https://a.example/p": PAGE})
    assert rows[0]["matched_by"] == "whole"
