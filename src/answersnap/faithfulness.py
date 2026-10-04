"""Is the quoted text actually on the page the engine attributed it to?

Only an engine that returns the quoted text (`cited_text`) can be checked: that
quote is the engine's own claim that this sentence came from this page. Engines
that return a link without a quote are reported as not observable — scoring
them would be scoring nothing, and a 0 would read as fabrication.

There is deliberately no pass/fail threshold. A row says "found on the page
verbatim", or "not found verbatim" with the best overlap score; pages change,
render through JS and sit behind paywalls, so a low score is a lead to check,
not a verdict.
"""

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from importlib import resources

from answersnap import store
from answersnap.fetch import safe_fetch
from answersnap.fetch.robots import RobotsCache
from answersnap.metrics.rates import is_placeholder
from answersnap.metrics.retrieval_gap import METHOD_VERSION as COMPARE_METHOD
from answersnap.metrics.retrieval_gap import compare, control_rows

# ---------------------------------------------------------------- constants
# Bump when text extraction or quote normalisation changes; never mix rows
# from two methods. html-text-1 put a space at every tag boundary, so
# "<a>Brand A Co.</a>, which" became "Brand A Co. , which" and real quotes
# never matched — found on the first live run, invisible on hand-written fixtures.
TEXT_METHOD = "html-text-2"
QUOTE_METHOD = "md-quote-2"
METHOD_VERSION = f"{TEXT_METHOD}+{QUOTE_METHOD}+{COMPARE_METHOD}"
SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg"}
# Only these separate words. Inline tags (a, b, em, span...) sit inside a
# sentence, and a space there breaks verbatim matching.
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
              "tr", "td", "th", "table", "thead", "tbody", "section", "article", "header",
              "footer", "nav", "main", "aside", "blockquote", "pre", "dd", "dt", "dl",
              "figure", "figcaption", "hr", "title", "option"}
# Engines quote pages as Markdown: headings, emphasis, list markers, links,
# and a trailing ellipsis where the quote was cut. The page has none of that.
MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
MD_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
MD_EMPHASIS = re.compile(r"(\*\*|__|`)")
TRAILING_ELLIPSIS = re.compile(r"(?:\.\.\.|…)\s*$")
# Engines also stitch a quote from separate places on one page — typically the
# page title, a line break, then a passage further down. md-quote-1 required
# the whole quote to be contiguous and so missed every such quote.
QUOTE_PIECE_BREAK = re.compile(r"\n+")
MATCHED_WHOLE = "whole"
MATCHED_PIECES = "pieces"
FIXTURE_PAGES = ("answersnap.examples", "fixtures/pages")
STATUS_OK = "ok"
STATUS_NOT_OBSERVABLE = "not_observable"
STATUS_NOT_CHECKED = "not_checked"
STATUS_NOT_CHECKABLE = "not_checkable"
STATUS_NO_CITATIONS = "no_citations"
# Outcomes that will not change on a retry, so they are cached on disk.
# Timeouts and server errors are not: `answersnap report` should try again.
DEFINITIVE_FETCH_STATUSES = ("ok", "gone", "auth_required", "paywalled", "robots_blocked",
                             "blocked_unsafe", "invalid_url", "not_in_dry_run")
RESULT_FOUND = "found"
RESULT_NOT_FOUND = "not_found"
RESULT_NOT_CHECKABLE = "not_checkable"


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skipping = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIPPED_TAGS:
            self._skipping += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIPPED_TAGS and self._skipping:
            self._skipping -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skipping:
            self.parts.append(data)


def visible_text(body):
    """Page text as a reader sees it: tags gone, entities decoded, blocks
    separated, inline markup joined without a gap."""
    if not body:
        return ""
    parser = _VisibleText()
    parser.feed(body)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def _comparable(text):
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def quote_pieces(quote):
    """The quote split where the engine broke lines, each piece normalised."""
    pieces = [normalise_quote(part) for part in QUOTE_PIECE_BREAK.split(quote or "")]
    return [piece for piece in pieces if piece]


def normalise_quote(quote):
    """The quote's words without the Markdown the engine wrapped them in."""
    text = MD_LINK.sub(r"\1", quote or "")
    text = MD_HEADING.sub("", text)
    text = MD_LIST_MARKER.sub("", text)
    text = MD_EMPHASIS.sub("", text)
    text = TRAILING_ELLIPSIS.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


class LiveFetcher:
    """Fetches through the SSRF guard, after asking robots.txt."""

    mode = "live"

    def __init__(self, robots=None):
        self._robots = robots or RobotsCache()

    def __call__(self, url):
        # URLs come from engine output; a malformed one (bad port, broken IPv6
        # literal) must cost one row, not the whole report.
        try:
            if not self._robots.allows(url):
                return "robots_blocked", None, None
            return safe_fetch.fetch(url)
        except ValueError:
            return "invalid_url", None, None


class FixtureFetcher:
    """Serves bundled pages for a dry run; everything else is 'not in dry run'."""

    mode = "dry_run"

    def __init__(self):
        root = resources.files(FIXTURE_PAGES[0]).joinpath(FIXTURE_PAGES[1])
        self._root = root
        self._pages = json.loads(root.joinpath("index.json").read_text(encoding="utf-8"))["pages"]

    def __call__(self, url):
        name = self._pages.get(url)
        if name is None:
            return "not_in_dry_run", None, None
        return "ok", 200, self._root.joinpath(name).read_text(encoding="utf-8")


def _cached_fetch(run_dir, url, fetcher, memo, clock):
    if url in memo:
        return memo[url]
    path = store.fetched_path(run_dir, url)
    page = store.read_json(path) if path.is_file() else None
    # Text extracted by another method version must not be scored under this
    # one; fetch again rather than mix methods.
    if page is None or page.get("text_method") != TEXT_METHOD:
        status, http_status, body = fetcher(url)
        page = {"url": url, "fetch_status": status, "http_status": http_status,
                "fetched_at": clock().isoformat(), "text_method": TEXT_METHOD,
                "text": visible_text(body) if status == "ok" else None}
        if status in DEFINITIVE_FETCH_STATUSES:
            store.write_json(path, page)
    memo[url] = page
    return page


def _row(record, citation, page, reason):
    row = {"engine": record["engine"], "query_index": record["query_index"],
           "repeat": record["repeat"], "url": citation["url"], "domain": citation.get("domain"),
           "cited_text": citation.get("cited_text"),
           "fetch_status": page["fetch_status"] if page else None,
           "method_version": METHOD_VERSION, "not_comparable_reason": reason,
           "exact_match": None, "normalized_match": None, "best_window_score": None}
    if reason is None:
        # The raw quote stays in the row for the reader; matching uses its words.
        row.update(compare(normalise_quote(citation["cited_text"]), page["text"]))
        row.update(_piecewise(citation["cited_text"], page["text"], row["normalized_match"]))
    row["result"] = _result(row)
    return row


def _piecewise(quote, page_text, whole_found):
    """Found only if every piece of the quote is on the page verbatim. Pieces
    need not be adjacent, but none may be missing: that is still the engine's
    own claim, checked word for word."""
    if whole_found:
        return {"matched_by": MATCHED_WHOLE, "pieces": 1}
    pieces = quote_pieces(quote)
    page = _comparable(page_text)
    found = sum(1 for piece in pieces if _comparable(piece) in page)
    if len(pieces) > 1 and found == len(pieces):
        return {"normalized_match": True, "matched_by": MATCHED_PIECES, "pieces": len(pieces)}
    return {"matched_by": None, "pieces": len(pieces), "pieces_found": found}


def _result(row):
    if row["not_comparable_reason"]:
        return RESULT_NOT_CHECKABLE
    return RESULT_FOUND if row["normalized_match"] else RESULT_NOT_FOUND


def _reason(citation, page):
    if not citation.get("cited_text"):
        return "no_cited_text"
    if page is None:
        return "not_fetched"
    if page["fetch_status"] != "ok":
        return f"fetch_{page['fetch_status']}"
    if not page["text"]:
        return "no_page_text"
    return None


def check(run_dir, records, fetcher=None, clock=None):
    """One row per cited source. fetcher=None means fetching was switched off."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    memo, rows = {}, []
    for record in records:
        if is_placeholder(record):
            continue
        for citation in record.get("citations", []):
            if not citation.get("is_cited"):
                continue
            page = None
            if citation.get("cited_text") and fetcher is not None:
                page = _cached_fetch(run_dir, citation["url"], fetcher, memo, clock)
            rows.append(_row(record, citation, page, _reason(citation, page)))
    return rows


def summarise(rows, engines, fetched=True):
    """Per-engine counts. An engine that never returned a quote is not
    observable — it is not 'zero faithful citations'."""
    summary = {}
    for engine in engines:
        mine = [r for r in rows if r["engine"] == engine]
        quoted = [r for r in mine if r["cited_text"]]
        found = sum(1 for r in mine if r["result"] == RESULT_FOUND)
        not_found = sum(1 for r in mine if r["result"] == RESULT_NOT_FOUND)
        summary[engine] = {
            "status": _summary_status(mine, quoted, found + not_found, fetched),
            "cited": len(mine), "with_quote": len(quoted),
            # The denominator for "found" is quotes we could actually compare;
            # uncheckable pages are counted apart, never as misses.
            "checked": found + not_found, "found": found, "not_found": not_found,
            "not_checkable": sum(1 for r in quoted if r["result"] == RESULT_NOT_CHECKABLE),
        }
    return summary


def _summary_status(mine, quoted, checked, fetched):
    if not mine:
        return STATUS_NO_CITATIONS
    if not quoted:
        return STATUS_NOT_OBSERVABLE
    if not fetched:
        return STATUS_NOT_CHECKED
    return STATUS_OK if checked else STATUS_NOT_CHECKABLE


def instrument_check():
    """Known-good quote/page pairs. If these fail, the tool is broken, and no
    'not found' result anywhere in the report can be trusted."""
    controls = control_rows()
    return {"method_version": METHOD_VERSION,
            "passed": all(r["exact_match"] for r in controls),
            "controls": [{"label": r["control_label"], "exact_match": r["exact_match"],
                          "best_window_score": r["best_window_score"]} for r in controls]}
