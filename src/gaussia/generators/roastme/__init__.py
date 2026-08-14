"""Roast Me — profile-then-exploit adversarial evaluation.

A generator subsystem rather than a metric (spec D8, D15): the Probe Library, the Profiler
and the Exploiter produce the Roast Dataset, and that dataset is what enters the metric
pipeline afterwards.

What is re-exported here is what a run needs and what costs nothing to import: none of it reaches
a probe engine, so none of it pulls a dependency of the ``roastme`` extra (FR-037). That rule is
the whole boundary — the three engines behind an extra are **not** re-exported, here or from
``probes``, and each is imported from its own module the way the framework already treats its
optional-backend adapters.

The rule had been applied to the entry points and to nothing else, which left the far end of a run
reachable only by naming a module. ``to_dataset`` is the single bridge to the metric pipeline
(FR-034) and lived at ``gaussia.generators.roastme.dataset``, while ``Profiler`` sat on the facade:
a user could start a run from the front door and not finish one without reading the source. The
same held for ``validate_catalogue``, which FR-025 expects to run *before* generation, and for the
scoring functions a report has to be re-derived from.

The three shipped collaborators the ``Exploiter`` constructor **requires** —
``PromptedQueryGenerator``, ``JudgeOnProfileFilter``, ``EmbeddingRealismEstimator`` — are here for
the same reason and were the last place the original complaint stayed literally true: a user could
reach ``Exploiter`` from the front door and still not construct one. None of the three touches an
engine, so the boundary is unchanged.

``gaussia.generators.__init__`` imports eagerly, so nothing here may be registered there.
"""

from .dataset import charged_principles, grading_methods, report_to_dataset, to_dataset, to_record, to_records
from .exploiter import Exploiter
from .probes.catalogue import validate_catalogue
from .probes.enumeration import EnumerationProbeEngine
from .probes.library import ProbeLibrary
from .probes.mentions import CompoundTokenExtractor, MentionExtractor
from .probes.transforms import TRANSFORMS, available, resolve
from .probes.verification import NearMissVerifier, collision
from .profiler import Profiler
from .searches.attribute_iteration import AttributeIterationSearch
from .searches.on_profile import JudgeOnProfileFilter
from .searches.query_generation import PromptedQueryGenerator
from .searches.realism import EmbeddingRealismEstimator
from .searches.scoring import category_score, standard_error, violation_score

__all__ = [
    "TRANSFORMS",
    "AttributeIterationSearch",
    "CompoundTokenExtractor",
    "EmbeddingRealismEstimator",
    "EnumerationProbeEngine",
    "Exploiter",
    "JudgeOnProfileFilter",
    "MentionExtractor",
    "NearMissVerifier",
    "ProbeLibrary",
    "Profiler",
    "PromptedQueryGenerator",
    "available",
    "category_score",
    "charged_principles",
    "collision",
    "grading_methods",
    "report_to_dataset",
    "resolve",
    "standard_error",
    "to_dataset",
    "to_record",
    "to_records",
    "validate_catalogue",
    "violation_score",
]
