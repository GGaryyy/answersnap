"""SSRF defences for a fetcher pointed at attacker-chosen URLs.

Two bypasses get explicit tests because they are the ones real implementations
get wrong: resolving twice (rebinding) and trusting the first hop (redirects).
"""

import ipaddress
import socket

import pytest

from answersnap.fetch.safe_fetch import (
    UnsafeUrlError,
    classify_address,
    fetch,
    resolve_and_check,
)

PUBLIC = "93.184.216.34"


def _resolver(*ips):
    """Serve one answer per call, so rebinding can be simulated."""
    answers = list(ips)

    def resolve(host, port):
        current = answers.pop(0) if len(answers) > 1 else answers[0]
        family = 10 if ":" in current else 2
        return [(family, 1, 6, "", (current, port))]
    return resolve


@pytest.mark.parametrize("ip,category", [
    ("127.0.0.1", "loopback"),
    ("169.254.169.254", "link_local"),
    ("10.0.0.5", "private"),
    ("192.168.1.1", "private"),
    ("172.16.0.1", "private"),
    ("224.0.0.1", "multicast"),
    ("0.0.0.0", "reserved"),
    ("::1", "loopback"),
    ("fe80::1", "link_local"),
    ("fd00::1", "private"),
    # IPv4-mapped IPv6: checking only the v4 form would miss this entirely.
    ("::ffff:10.0.0.1", "private"),
    ("::ffff:169.254.169.254", "link_local"),
    # CGNAT: cloud providers route internally over it, and it is neither
    # private nor reserved by name — a named blacklist misses it entirely.
    ("100.64.0.1", "not_global"),
])
def test_non_public_addresses_are_classified(ip, category):
    assert classify_address(ipaddress.ip_address(ip)) == category


@pytest.mark.parametrize("ip", [PUBLIC, "2606:2800:220:1::1"])
def test_public_addresses_pass(ip):
    assert classify_address(ipaddress.ip_address(ip)) is None


def test_the_block_reason_says_which_kind_of_address():
    with pytest.raises(UnsafeUrlError) as excinfo:
        resolve_and_check("https://evil.test/x", resolver=_resolver("169.254.169.254"))
    # link_local is almost never a legitimate citation; private shows up in
    # honest writing about networking. The follow-up differs, so the reason
    # has to survive.
    assert excinfo.value.category == "link_local"


def test_a_host_answering_with_both_public_and_private_is_refused():
    def mixed(host, port):
        return [(2, 1, 6, "", (PUBLIC, port)), (2, 1, 6, "", ("10.0.0.1", port))]

    with pytest.raises(UnsafeUrlError, match="private"):
        resolve_and_check("https://evil.test/x", resolver=mixed)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x.test/",
                                 "ftp://x.test/", "data:text/plain,hi"])
def test_only_http_schemes_are_allowed(url):
    with pytest.raises(UnsafeUrlError):
        resolve_and_check(url, resolver=_resolver(PUBLIC))


def test_an_unresolvable_host_is_refused_not_attempted():
    def broken(host, port):
        raise socket.gaierror("nope")

    with pytest.raises(UnsafeUrlError, match="does not resolve"):
        resolve_and_check("https://nx.test/x", resolver=broken)


def test_we_connect_to_the_address_we_checked():
    """The rebinding defence: one resolution, and it is the one we dial."""
    dialled = []

    def opener(url, address):
        dialled.append(str(address))
        return 200, None, "內容"

    # Second answer is private: a client that resolved again would land there.
    resolver = _resolver(PUBLIC, "10.0.0.7")
    status, http_status, body = fetch("https://a.test/x", resolver=resolver,
                                      opener=opener)
    assert (status, http_status, body) == ("ok", 200, "內容")
    assert dialled == [PUBLIC]


def test_a_redirect_into_the_metadata_service_is_refused():
    hops = []

    def opener(url, address):
        hops.append(url)
        if len(hops) == 1:
            return 302, "http://169.254.169.254/latest/meta-data/", None
        return 200, None, "secrets"

    status, _, reason = fetch("https://a.test/x",
                              resolver=lambda host, port: _resolver(
                                  PUBLIC if host == "a.test" else "169.254.169.254")(host, port),
                              opener=opener)
    # The first URL was clean; only revalidating the hop catches this.
    assert status == "blocked_unsafe"
    assert reason == "link_local"
    assert len(hops) == 1


def test_a_legitimate_redirect_is_followed():
    def opener(url, address):
        if url.endswith("/old"):
            return 301, "https://a.test/new", None
        return 200, None, "新頁面"

    status, http_status, body = fetch("https://a.test/old", resolver=_resolver(PUBLIC),
                                      opener=opener)
    assert (status, http_status, body) == ("ok", 200, "新頁面")


def test_a_redirect_loop_ends_rather_than_spinning():
    def opener(url, address):
        return 302, "https://a.test/loop", None

    status, _, _ = fetch("https://a.test/loop", resolver=_resolver(PUBLIC),
                         opener=opener, max_redirects=2)
    assert status == "http_error"


def test_a_network_error_is_not_reported_as_a_deletion():
    def opener(url, address):
        raise OSError("connection reset")

    status, _, _ = fetch("https://a.test/x", resolver=_resolver(PUBLIC), opener=opener)
    assert status == "http_error"


def test_a_timeout_has_its_own_status():
    def opener(url, address):
        raise TimeoutError()

    status, _, _ = fetch("https://a.test/x", resolver=_resolver(PUBLIC), opener=opener)
    assert status == "timeout"


@pytest.mark.parametrize("code,expected", [
    (200, "ok"), (204, "ok"), (401, "auth_required"), (403, "auth_required"),
    (402, "paywalled"), (404, "gone"), (410, "gone"), (500, "http_error"),
])
def test_status_codes_keep_their_distinct_meanings(code, expected):
    def opener(url, address):
        return code, None, "body"

    status, http_status, _ = fetch("https://a.test/x", resolver=_resolver(PUBLIC),
                                   opener=opener)
    assert status == expected
    assert http_status == code
