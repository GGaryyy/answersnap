"""answersnap must never import the hosting project it currently lives inside.

It is meant to move to its own public repository; one `from src...` import
would make that move break, and would ship a dependency nobody can install.
"""

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "answersnap"
FORBIDDEN = re.compile(r"^\s*(from\s+src(\.|\s)|import\s+src(\.|\s|$))", re.MULTILINE)


def test_no_module_imports_the_host_project():
    offenders = [p.relative_to(PACKAGE_ROOT).as_posix()
                 for p in PACKAGE_ROOT.rglob("*.py") if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_package_ships_the_files_the_runtime_reads():
    for relative in ("providers/recorded/anthropic.json", "providers/recorded/openai.json",
                     "providers/recorded/google.json", "report/templates/report.html.j2",
                     "examples/example.yaml", "examples/fixtures/pages/index.json"):
        assert (PACKAGE_ROOT / relative).is_file(), relative
