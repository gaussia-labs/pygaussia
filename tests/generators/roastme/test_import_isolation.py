"""Import isolation with the `roastme` extra uninstalled (T028, FR-037, SC-011).

The extra is `["sentence-transformers", "torch", "networkx"]`, so those three are what a user who
has not installed it does not have. The check runs in a subprocess with a meta-path finder that
refuses them, which is the only way to observe absence in a development environment that has them
installed.

Asserted per module rather than through the `gaussia.core` package facade on purpose. Importing
`gaussia.core` pulls numpy, because `core/embedder.py` imports it at module level and is
re-exported; that is pre-existing and harmless, since numpy arrives with the mandatory `scipy` and
`transformers` dependencies. What SC-011 requires is that each of the ten Roast Me interfaces, the
schemas and the subsystem facade import without the three the extra adds — so those three, and
only those, are blocked.

The last case is the one the plan warns about: `probes/__init__.py` must not re-export an engine,
or `from gaussia.generators.roastme import Profiler` would pull an embedder and break FR-037 one
level below where the file tables guard against it.
"""

import subprocess
import sys
import textwrap

BLOCKED = ("sentence_transformers", "torch", "networkx")

INTERFACE_MODULES = (
    "gaussia.core.grader",
    "gaussia.core.probe_engine",
    "gaussia.core.entity_enumerator",
    "gaussia.core.hook_verifier",
    "gaussia.core.transform",
    "gaussia.core.target_assistant",
    "gaussia.core.query_generator",
    "gaussia.core.on_profile_filter",
    "gaussia.core.realism_estimator",
    "gaussia.core.category_search",
)

SUBSYSTEM_MODULES = (
    "gaussia.schemas.roastme",
    "gaussia.generators.roastme",
    "gaussia.generators.roastme.probes",
    "gaussia.generators.roastme.searches",
    "gaussia.generators.roastme.searches.scoring",
    "gaussia.generators.roastme.searches.thresholds",
    "gaussia.generators.roastme.searches.policy_gradient",
    "gaussia.generators.roastme.profiler",
    "gaussia.generators.roastme.dataset",
    "gaussia.graders.logprob",
)

_SCRIPT = textwrap.dedent(
    """
    import importlib
    import sys

    BLOCKED = {blocked!r}
    TARGETS = {targets!r}
    FORBIDDEN_SUBMODULES = {forbidden!r}


    class Refuse:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                raise ImportError("the roastme extra is not installed: " + fullname)
            return None


    sys.meta_path.insert(0, Refuse())

    for name in TARGETS:
        importlib.import_module(name)

    loaded = [name for name in FORBIDDEN_SUBMODULES if name in sys.modules]
    if loaded:
        raise SystemExit("eagerly imported: " + ", ".join(loaded))
    """
)


def _import_in_isolation(targets, forbidden=()):
    script = _SCRIPT.format(blocked=BLOCKED, targets=tuple(targets), forbidden=tuple(forbidden))
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheInterfacesImportWithoutTheExtra:
    def test_all_ten(self):
        completed = _import_in_isolation(INTERFACE_MODULES)
        assert completed.returncode == 0, completed.stderr

    def test_one_at_a_time(self):
        for module in INTERFACE_MODULES:
            completed = _import_in_isolation([module])
            assert completed.returncode == 0, f"{module}\n{completed.stderr}"


class TestTheSubsystemImportsWithoutTheExtra:
    def test_the_schemas_and_the_facade(self):
        completed = _import_in_isolation(SUBSYSTEM_MODULES)
        assert completed.returncode == 0, completed.stderr

    def test_the_probe_package_re_exports_no_engine(self):
        """The engines are imported directly by the user, the way the framework already treats its
        numpy-dependent abstractions."""
        completed = _import_in_isolation(
            ["gaussia.generators.roastme"],
            forbidden=[
                "gaussia.generators.roastme.probes.retrieval",
                "gaussia.generators.roastme.probes.graph",
                "gaussia.generators.roastme.probes.grag",
            ],
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_the_policy_gradient_loop_imports_without_the_training_stack(self):
        """The loop and the weight update live in different modules so the CPU tests of the loop
        do not drag the `roastme-rl` stack into the default suite."""
        completed = _import_in_isolation(["gaussia.generators.roastme.searches.policy_gradient"])
        assert completed.returncode == 0, completed.stderr


class TestTheRestOfTheFrameworkStillImports:
    def test_importing_gaussia_succeeds_with_the_extra_blocked(self):
        completed = _import_in_isolation(["gaussia", "gaussia.core", "gaussia.schemas"])
        assert completed.returncode == 0, completed.stderr

    def test_the_blocker_actually_blocks(self):
        """Guards the guard: if the finder were a no-op every case above would pass vacuously."""
        completed = _import_in_isolation(["torch"])
        assert completed.returncode != 0
        assert "roastme extra is not installed" in completed.stderr
