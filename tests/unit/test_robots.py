from answersnap.fetch import robots


def _resolves_to(address):
    """A getaddrinfo stand-in returning one address."""
    return lambda host, port: [(None, None, None, "", (address, port))]


PUBLIC = _resolves_to("93.184.216.34")


def test_robots_txt_is_read_through_the_pinned_guarded_fetch(monkeypatch):
    seen = {}

    def fake_open(url, address):
        seen["url"], seen["address"] = url, address
        return 200, None, "User-agent: *\nDisallow: /private\n"
    monkeypatch.setattr(robots, "open_pinned", fake_open)
    parser = robots._read_robots("https://site.example", resolver=PUBLIC)
    assert seen["url"] == "https://site.example/robots.txt"
    assert str(seen["address"]) == "93.184.216.34"
    assert parser.can_fetch(robots.USER_AGENT, "https://site.example/private/x") is False
    assert parser.can_fetch(robots.USER_AGENT, "https://site.example/public") is True


def test_robots_on_an_internal_address_is_never_fetched(monkeypatch):
    def must_not_open(*args):
        raise AssertionError("opened a connection to an internal address")
    monkeypatch.setattr(robots, "open_pinned", must_not_open)
    assert robots._read_robots("http://internal.example", resolver=_resolves_to("10.0.0.5")) is None


def test_unreadable_robots_txt_is_permissive(monkeypatch):
    monkeypatch.setattr(robots, "open_pinned", lambda url, address: (404, None, None))
    assert robots._read_robots("https://site.example", resolver=PUBLIC) is None

    def dropped(url, address):
        raise OSError("connection reset")
    monkeypatch.setattr(robots, "open_pinned", dropped)
    assert robots._read_robots("https://site.example", resolver=PUBLIC) is None


def test_cache_asks_each_host_once_and_allows_when_unreadable():
    calls = []

    def opener(root, resolver=None):
        calls.append(root)
        return None
    cache = robots.RobotsCache(opener=opener)
    assert cache.allows("https://a.example/x") and cache.allows("https://a.example/y")
    assert cache.allows("https://b.example/")
    assert calls == ["https://a.example", "https://b.example"]


def test_a_hostile_robots_server_cannot_stop_the_report(monkeypatch):
    import http.client

    def broken(url, address):
        raise http.client.BadStatusLine("garbage")
    monkeypatch.setattr(robots, "open_pinned", broken)
    assert robots._read_robots("https://site.example", resolver=PUBLIC) is None


def test_server_error_on_robots_txt_means_disallow(monkeypatch):
    monkeypatch.setattr(robots, "open_pinned", lambda url, address: (503, None, None))
    parser = robots._read_robots("https://site.example", resolver=PUBLIC)
    assert parser.can_fetch(robots.USER_AGENT, "https://site.example/anything") is False
