"""Minimal JSON POST with backoff, for platforms without a first-party SDK.

Kept tiny and dependency-free on purpose: the sampling path must be easy to read
end to end, because everything it does silently becomes a measurement claim.
"""

import json
import random
import time
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 120
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2
# Five platforms on one schedule retry in lockstep without it, turning a
# recoverable delay into a synchronised wall of failures.
JITTER_FRACTION = 0.3
MAX_RESPONSE_BYTES = 8_000_000
RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504)


class HttpError(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _backoff_seconds(attempt, retry_after=None, rng=random.random):
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    base = BACKOFF_BASE_SECONDS * (2 ** attempt)
    return base * (1 + JITTER_FRACTION * rng())


def post_json(url, body, headers=None, timeout=TIMEOUT_SECONDS, sleep=time.sleep,
              rng=random.random):
    payload = json.dumps(body).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **(headers or {})}

    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(url, data=payload, headers=all_headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read(MAX_RESPONSE_BYTES).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # A rate limit that ends the call turns a recoverable delay into a
            # permanent hole in the time series.
            if exc.code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                # The platform telling us when to come back beats our guess.
                retry_after = (exc.headers or {}).get("Retry-After")
                sleep(_backoff_seconds(attempt, retry_after, rng))
                continue
            raise HttpError(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_RETRIES:
                sleep(_backoff_seconds(attempt, rng=rng))
                continue
            raise HttpError(f"{type(exc).__name__}") from exc
        except json.JSONDecodeError as exc:
            raise HttpError("response was not JSON") from exc

    raise HttpError("retries exhausted")
