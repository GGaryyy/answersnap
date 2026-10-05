"""Fetching URLs that an attacker may have chosen.

Citation URLs come out of AI answers, which is to say out of pages the model
retrieved — content anyone can publish. This module is the product's own threat
model turned on itself: a system that fetches attacker-influenced URLs is a
request forger unless it is careful, and being breached by the class of attack
we sell detection for is the worst outcome available to this project.

Two classic bypasses are handled explicitly:

  - DNS rebinding: resolving once for the check and letting the HTTP client
    resolve again for the connection lets a short-TTL record answer differently
    the second time. So the checked address is the address we connect to.
  - Redirects: a clean first URL that 302s to the metadata service bypasses a
    check done only on the original. So redirects are followed by hand, and
    every hop is revalidated from scratch.

When anything is unclear, the answer is "do not fetch".
"""

import http.client
import ipaddress
import socket
import ssl
from urllib.parse import quote, urljoin, urlparse

from answersnap import __version__

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_PORTS = {"http": 80, "https": 443}
TIMEOUT_SECONDS = 10
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
USER_AGENT = f"answersnap/{__version__} (+https://github.com/GGaryyy/answersnap)"


class UnsafeUrlError(ValueError):
    def __init__(self, reason, category="unsafe"):
        super().__init__(reason)
        self.reason = reason
        # link_local is almost never a legitimate citation; private and loopback
        # do appear in genuine writing about networking, so the categories are
        # kept apart rather than collapsed into one verdict.
        self.category = category


def classify_address(address):
    """Return why an address is off limits, or None if it is publicly routable.

    The decision is a whitelist — `is_global` — because a blacklist of named
    ranges keeps missing one. CGNAT (100.64.0.0/10), which cloud providers use
    for internal routing, is neither private nor reserved by Python's naming and
    would have slipped straight through an enumerated list.
    """
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    # Named categories first, and not only for the label: multicast addresses
    # report is_global True, so a whitelist alone would wave 224.0.0.0/4
    # straight through.
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link_local"
    if address.is_multicast:
        return "multicast"
    if address.is_unspecified or address.is_reserved:
        return "reserved"
    if address.is_private:
        return "private"
    # Backstop: anything the named checks miss but that is not publicly
    # routable — CGNAT (100.64.0.0/10) is the one that bites, since cloud
    # providers route internally over it and it is neither private nor reserved.
    if not address.is_global:
        return "not_global"
    return None


def resolve_and_check(url, resolver=None):
    """Return the addresses we are allowed to connect to for this URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"scheme '{parsed.scheme}' not allowed", "scheme")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("no host in url", "no_host")

    resolver = resolver or socket.getaddrinfo
    try:
        infos = resolver(host, parsed.port or DEFAULT_PORTS[parsed.scheme])
    except socket.gaierror as exc:
        raise UnsafeUrlError("host does not resolve", "unresolvable") from exc

    addresses = []
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError as exc:
            raise UnsafeUrlError("unparseable address", "unparseable") from exc
        category = classify_address(address)
        if category:
            # One bad answer condemns the name: a host that resolves to both a
            # public and a private address is the rebinding pattern itself.
            raise UnsafeUrlError(f"host resolves to a {category} address", category)
        addresses.append(address)

    if not addresses:
        raise UnsafeUrlError("host resolved to nothing", "unresolvable")
    return addresses


def open_pinned(url, address, timeout=TIMEOUT_SECONDS):
    """Connect to a specific address, presenting the original hostname.

    The address came from the check above, so no second resolution can happen
    between deciding and connecting.
    """
    parsed = urlparse(url)
    port = parsed.port or DEFAULT_PORTS[parsed.scheme]
    host_header = parsed.netloc
    # Non-ASCII paths must go out percent-encoded; the request line is latin-1.
    path = quote(parsed.path or "/", safe="/%:@&=+$,~*!'()")
    query = quote(parsed.query, safe="/%:@&=+$,~*!'()?") if parsed.query else ""
    target = f"{path}{'?' + query if query else ''}"
    literal = (f"[{address}]" if address.version == 6 else str(address))

    if parsed.scheme == "https":
        # Connect to the checked address, but keep validating the certificate
        # and sending SNI for the hostname we were actually asked for.
        connection = _https_pinned(parsed.hostname, literal, port, timeout)
    else:
        connection = http.client.HTTPConnection(literal, port, timeout=timeout)

    try:
        connection.putrequest("GET", target, skip_host=True, skip_accept_encoding=True)
        connection.putheader("Host", host_header)
        connection.putheader("User-Agent", USER_AGENT)
        connection.putheader("Accept-Encoding", "identity")  # no decompression bombs
        connection.endheaders()
        response = connection.getresponse()
        status = response.status
        location = response.getheader("Location")
        body = response.read(MAX_BYTES).decode("utf-8", "replace")
    finally:
        connection.close()
    return status, location, body


def _https_pinned(hostname, literal, port, timeout):
    """An HTTPS connection to a pinned IP that still validates the hostname."""
    context = ssl.create_default_context()

    class PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            self.sock = context.wrap_socket(
                socket.create_connection((literal, port), timeout),
                server_hostname=hostname)

    return PinnedHTTPSConnection(hostname, port, timeout=timeout, context=context)


def fetch(url, *, resolver=None, opener=None, max_redirects=MAX_REDIRECTS):
    """Fetch a URL, revalidating every redirect hop.

    Returns (status, http_status, text). `status` is a `fetch_status` value.
    """
    opener = opener or open_pinned
    current = url
    for _ in range(max_redirects + 1):
        try:
            addresses = resolve_and_check(current, resolver=resolver)
        except UnsafeUrlError as exc:
            return "blocked_unsafe", None, exc.category

        try:
            http_status, location, body = opener(current, addresses[0])
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            if isinstance(exc, (TimeoutError, socket.timeout)):
                return "timeout", None, None
            return "http_error", None, None

        if http_status in (301, 302, 303, 307, 308) and location:
            # Re-enter the loop so the next hop is checked from scratch.
            current = urljoin(current, location)
            continue
        if http_status in (401, 403):
            return "auth_required", http_status, None
        if http_status == 402:
            return "paywalled", http_status, None
        if http_status in (404, 410):
            # The only codes that mean the source is actually gone. A 500 or a
            # dropped connection means we could not read it, which must never
            # become "it was deleted".
            return "gone", http_status, None
        if 200 <= http_status < 300:
            return "ok", http_status, body
        return "http_error", http_status, None

    return "http_error", None, None
