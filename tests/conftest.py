"""answersnap's own suite: no database, no network, no API keys.

This directory has no __init__.py on purpose. pytest then puts it on sys.path,
so helpers import as `fake_answers` rather than `tests.fake_answers` — the host
repo also has a `tests` package, and the two must not collide while this
package still lives inside it.
"""
