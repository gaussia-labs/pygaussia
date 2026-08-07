"""Roast Me — profile-then-exploit adversarial evaluation.

A generator subsystem rather than a metric (spec D8, D15): the Probe Library, the Profiler
and the Exploiter produce the Roast Dataset, and that dataset is what enters the metric
pipeline afterwards.

The three entry points are re-exported here because none of them imports a probe engine, so
none pulls a dependency of the ``roastme`` extra (FR-037). The engines themselves are **not**
re-exported, here or from ``probes``: each is imported from its own module, the way the
framework already treats its optional-backend adapters, so installing one extra is enough.

``gaussia.generators.__init__`` imports eagerly, so nothing here may be registered there.
"""

from .exploiter import Exploiter
from .probes.library import ProbeLibrary
from .profiler import Profiler

__all__ = ["Exploiter", "ProbeLibrary", "Profiler"]
