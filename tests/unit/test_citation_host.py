"""Which site a citation points at — wrong here, and every tally is wrong.

Gemini returns every source through a Google redirect with the real domain in
the title. Read naively, every clinic's site is "cited 0 times" on Gemini, and
the letters would print that as a finding.
"""

from answersnap.providers.citation_host import (
    citation_host,
    is_redirector,
    resolve_domain,
    strip_www,
)

REDIRECT = ("https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
            "AUZIYQHNSYjOX5ttuhfh01W")


def test_a_google_redirect_resolves_to_the_domain_in_its_title():
    assert resolve_domain(REDIRECT, "smile-dental.tw") == "smile-dental.tw"
    assert citation_host({"url": REDIRECT, "title": "www.trlsd.example"}) == "trlsd.example"


def test_a_redirect_whose_title_is_not_a_host_stays_unresolved():
    # A title like "植牙推薦 2026" is a page title, not a site. Guessing would
    # attribute the citation to nobody in particular — better to keep the
    # redirector visible than to invent a domain.
    host = resolve_domain(REDIRECT, "植牙推薦 2026")
    assert host == "vertexaisearch.cloud.google.com"


def test_ordinary_urls_are_untouched_except_for_www():
    dcard = {"url": "https://www.dcard.tw/f/teeth", "title": "x"}
    king = {"url": "https://guide-clinic.example/guide", "title": None}
    assert citation_host(dcard) == "dcard.tw"
    assert citation_host(king) == "guide-clinic.example"


def test_a_stored_domain_wins_over_re_deriving_it():
    # Records written by the sampler carry the adapter's resolved domain; a
    # reader must trust it rather than re-parse a URL it may not understand.
    c = {"url": REDIRECT, "title": "noise", "domain": "renhodental.com"}
    assert citation_host(c) == "renhodental.com"


def test_strip_www_removes_a_prefix_not_a_character_set():
    assert strip_www("www.wallace.tw") == "wallace.tw"
    assert strip_www("wallace.tw") == "wallace.tw"  # lstrip would give allace.tw
    assert strip_www("web.example.com") == "web.example.com"


def test_redirector_detection():
    assert is_redirector({"url": REDIRECT})
    assert not is_redirector({"url": "https://trlsd.example/"})
