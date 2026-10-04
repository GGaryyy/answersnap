"""robots.txt lookups, fetched through the same guard as every page fetch."""

import http.client
import urllib.robotparser
from urllib.parse import urlparse, urlunparse

from answersnap.fetch.safe_fetch import USER_AGENT, UnsafeUrlError, open_pinned, resolve_and_check

# ---------------------------------------------------------------- constants
# RFC 9309: a server error on robots.txt means "assume disallowed"; a 4xx means
# there are no rules.
SERVER_ERROR_MIN = 500


class RobotsCache:
    """robots.txt lookups, fetched through the same guard as everything else.

    Reaching robots.txt means making a request to a host named in a citation —
    which is to say, a host an attacker may have chosen. Fetching it outside the
    address check would leave a blind SSRF wide open right next to the one we
    closed.
    """

    def __init__(self, opener=None, resolver=None):
        self._parsers = {}
        self._opener = opener or _read_robots
        self._resolver = resolver

    def allows(self, url):
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._parsers:
            self._parsers[root] = self._opener(root, self._resolver)
        parser = self._parsers[root]
        if parser is None:  # robots.txt unreachable: treat as permissive
            return True
        return parser.can_fetch(USER_AGENT, url)


def _read_robots(root, resolver=None):
    robots_url = urlunparse((*urlparse(root)[:2], "/robots.txt", "", "", ""))
    try:
        addresses = resolve_and_check(robots_url, resolver=resolver)
    except UnsafeUrlError:
        return None
    try:
        http_status, _, body = open_pinned(robots_url, addresses[0])
    except (OSError, http.client.HTTPException):
        # The host is hostile or broken; either way it cannot be allowed to
        # stop the report. Unreachable counts as no rules.
        return None
    parser = urllib.robotparser.RobotFileParser()
    if http_status >= SERVER_ERROR_MIN:
        parser.disallow_all = True
        return parser
    if http_status != 200 or body is None:
        return None
    parser.parse(body.splitlines())
    return parser
