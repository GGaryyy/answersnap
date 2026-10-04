"""answersnap: auditable snapshots of what AI answer engines say about a brand."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("answersnap")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0.0.0+unknown"
