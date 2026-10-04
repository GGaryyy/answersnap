"""Retry behaviour for the platforms without a first-party SDK.

A rate limit that ends the call turns a recoverable delay into a permanent hole
in the time series — and a hole is indistinguishable from the brand not being
mentioned. That makes this retry loop a measurement concern, not plumbing.
"""

import json
import urllib.error

import pytest

from answersnap.providers.http import MAX_RETRIES, HttpError, post_json


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self, limit=None):
        return self._body[:limit] if limit else self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code):
    error = urllib.error.HTTPError("https://x.test", code, "err", {}, None)
    error.headers = {}
    return error


class Sequence:
    """Serves a scripted list of outcomes, one per attempt."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.attempts = 0

    def __call__(self, request, timeout=None):
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


@pytest.fixture
def slept():
    delays = []
    return delays, delays.append


def test_a_rate_limit_is_retried_rather_than_lost(monkeypatch, slept):
    delays, sleep = slept
    transport = Sequence([_http_error(429), _http_error(429), {"ok": True}])
    monkeypatch.setattr("urllib.request.urlopen", transport)

    assert post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0) == {"ok": True}
    assert transport.attempts == 3
    # Backoff grows, so we are not hammering the platform we are measuring.
    assert delays == sorted(delays) and len(delays) == 2


def test_the_platforms_own_retry_after_wins_over_our_guess(monkeypatch, slept):
    delays, sleep = slept
    error = _http_error(429)
    error.headers = {"Retry-After": "42"}
    transport = Sequence([error, {"ok": True}])
    monkeypatch.setattr("urllib.request.urlopen", transport)

    post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0)
    assert delays == [42.0]


def test_backoff_is_jittered(monkeypatch, slept):
    delays, sleep = slept
    transport = Sequence([_http_error(503), _http_error(503), {"ok": True}])
    monkeypatch.setattr("urllib.request.urlopen", transport)

    post_json("https://x.test", {}, sleep=sleep, rng=lambda: 1.0)
    plain = [2, 4]
    # Without jitter five platforms on one schedule retry in lockstep.
    assert all(actual > nominal for actual, nominal in zip(delays, plain))


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(monkeypatch, slept, code):
    _, sleep = slept
    transport = Sequence([_http_error(code), {"ok": True}])
    monkeypatch.setattr("urllib.request.urlopen", transport)
    assert post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0) == {"ok": True}


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_statuses_fail_immediately(monkeypatch, slept, code):
    _, sleep = slept
    transport = Sequence([_http_error(code)])
    monkeypatch.setattr("urllib.request.urlopen", transport)

    with pytest.raises(HttpError, match=f"HTTP {code}"):
        post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0)
    # Retrying a 401 just burns time and money.
    assert transport.attempts == 1


def test_retries_are_bounded(monkeypatch, slept):
    _, sleep = slept
    transport = Sequence([_http_error(429)] * (MAX_RETRIES + 1))
    monkeypatch.setattr("urllib.request.urlopen", transport)

    with pytest.raises(HttpError, match="HTTP 429"):
        post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0)
    assert transport.attempts == MAX_RETRIES + 1


def test_connection_errors_are_retried_then_surfaced(monkeypatch, slept):
    _, sleep = slept
    transport = Sequence([urllib.error.URLError("boom")] * (MAX_RETRIES + 1))
    monkeypatch.setattr("urllib.request.urlopen", transport)

    with pytest.raises(HttpError, match="URLError"):
        post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0)


def test_a_non_json_body_is_reported_not_guessed(monkeypatch, slept):
    _, sleep = slept

    class NotJson(FakeResponse):
        def read(self, limit=None):
            return b"<html>rate limited</html>"

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda request, timeout=None: NotJson({}))
    with pytest.raises(HttpError, match="not JSON"):
        post_json("https://x.test", {}, sleep=sleep, rng=lambda: 0.0)


def test_the_request_carries_the_body_and_headers(monkeypatch, slept):
    _, sleep = slept
    captured = {}

    def transport(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = request.headers
        return FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", transport)
    post_json("https://x.test", {"model": "sonar"},
              headers={"Authorization": "Bearer k"}, sleep=sleep, rng=lambda: 0.0)

    assert captured["body"] == {"model": "sonar"}
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["headers"]["Content-type"] == "application/json"
