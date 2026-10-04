"""Which site a citation points at — the one question every tally asks.

Two things make the naive `urlparse(url).netloc` wrong here:

- Gemini's grounding returns every source as a redirect through
  vertexaisearch.cloud.google.com. The real site is in the chunk's `title`
  (Google fills it with the bare domain). Counted naively, every clinic's site
  is "cited 0 times" on Gemini — a measurement artefact wearing the shape of
  a finding.
- `www.` is a prefix, not part of the identity. The domain map says
  www.clinic.example; Gemini says clinic.example; both are the same site.

Used by every reader of survey answer records, so the answer to "is this the
clinic's own site" is the same in the ranking, the letters and the evidence
sheet.
"""

import re
from urllib.parse import urlparse

# ---------------------------------------------------------------- constants
GOOGLE_REDIRECTOR = "vertexaisearch.cloud.google.com"
HOST_SHAPE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$")


def strip_www(domain):
    """`lstrip("www.")` strips a character SET, not a prefix.

    It turns wallace.example into allace.example and web.example.com into
    eb.example.com. When the two sides of a comparison get eaten differently,
    an owned domain stops matching its own citation and the rate falls to zero —
    a measurement failure wearing the shape of "nobody cites you".
    """
    domain = (domain or "").lower()
    return domain[4:] if domain.startswith("www.") else domain


def looks_like_host(text):
    return bool(text) and HOST_SHAPE.match(text.strip().lower()) is not None


def resolve_domain(url, title=None):
    """The site a URL really points at, seeing through Google's redirector."""
    host = (urlparse(url or "").netloc or "").lower()
    if host == GOOGLE_REDIRECTOR and looks_like_host(title):
        return title.strip().lower()
    return host


def citation_host(citation):
    """Normalised host for one stored citation record ({url, title, domain?})."""
    domain = citation.get("domain") or resolve_domain(citation.get("url"),
                                                      citation.get("title"))
    return strip_www(domain)


def is_redirector(citation):
    host = (urlparse(citation.get("url") or "").netloc or "").lower()
    return host == GOOGLE_REDIRECTOR
