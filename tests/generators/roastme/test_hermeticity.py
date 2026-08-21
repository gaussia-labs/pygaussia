"""The default Roast Me suite is hermetic (T029, SC-010).

Enforced rather than asserted by convention. Two mechanisms, because either alone leaves a hole:

* the collected items are inspected for the markers that mean "this needs something the default
  suite does not have". Anything needing a GPU or a credential has to carry one, and no Roast Me
  test may carry one — otherwise the suite silently stops being runnable on any machine;
* the test sources are inspected for a network client or a credential read, which is what a test
  would need before it could carry the wrong marker in the first place.

The markers are also checked for being declared in the pytest configuration, because
`--strict-markers` turns an undeclared marker into a collection error rather than into a silent
no-op — which is what makes the first mechanism enforcement.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

ROAST_ME_TEST_DIRECTORIES = (
    ROOT / "tests" / "generators" / "roastme",
    ROOT / "tests" / "graders",
    ROOT / "tests" / "fixtures" / "roastme",
)

OFF_LIMITS_MARKERS = ("requires_gpu", "requires_api_key")

NETWORK_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:socket|ssl|requests|httpx|urllib|http|aiohttp|openai|anthropic|boto3)\b",
    re.MULTILINE,
)
CREDENTIAL_READ = re.compile(r"os\.environ|os\.getenv|getenv\(")


def _sources() -> list[Path]:
    return sorted(path for directory in ROAST_ME_TEST_DIRECTORIES for path in directory.rglob("*.py"))


def _roast_me_items(session) -> list:
    prefixes = tuple(str(directory) for directory in ROAST_ME_TEST_DIRECTORIES)
    return [item for item in session.items if str(item.path.resolve()).startswith(prefixes)]


class TestTheMarkersAreEnforceable:
    def test_every_off_limits_marker_is_declared(self):
        """An undeclared marker would be a collection error under `--strict-markers`, so declaring
        them is what lets a test that genuinely needs a GPU say so."""
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = {entry.split(":")[0].strip() for entry in config["tool"]["pytest"]["ini_options"]["markers"]}

        for marker in OFF_LIMITS_MARKERS:
            assert marker in declared

    def test_strict_markers_is_on(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert "--strict-markers" in config["tool"]["pytest"]["ini_options"]["addopts"]


class TestNoCollectedRoastMeTestNeedsAnythingSpecial:
    def test_the_suite_collected_something(self, request):
        assert _roast_me_items(request.session) != []

    @pytest.mark.parametrize("marker", OFF_LIMITS_MARKERS)
    def test_no_item_carries_it(self, request, marker):
        carriers = [item.nodeid for item in _roast_me_items(request.session) if item.get_closest_marker(marker)]
        assert carriers == []


class TestNoSourceReachesOutside:
    def test_the_scan_covers_every_module(self):
        paths = _sources()
        assert len(paths) >= 3
        assert all(path.exists() for path in paths)

    def test_no_network_client_is_imported(self):
        offenders = [
            str(path.relative_to(ROOT))
            for path in _sources()
            if NETWORK_IMPORT.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_no_credential_is_read(self):
        offenders = [
            str(path.relative_to(ROOT))
            for path in _sources()
            if CREDENTIAL_READ.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_no_url_is_contacted(self):
        offenders = [
            str(path.relative_to(ROOT))
            for path in _sources()
            if re.search(r"https?://", path.read_text(encoding="utf-8"))
        ]
        assert offenders == []
