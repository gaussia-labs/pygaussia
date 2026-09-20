"""Graders: the estimators of ``pi_hat_j`` bound to the principles of a behavioral contract.

The interface lives in ``gaussia.core.grader``; what ships here are reference implementations
for users who do not want to write their own (FR-007). Graders are substitutable without
touching anything downstream, so nothing outside a contract may depend on which one ran —
the grade records that instead (FR-005).

``LogprobGrader`` reads a verdict out of a judge model. ``ToolScopeGrader`` reads one out of
a set difference and needs no model at all.
"""

from .logprob import LOGPROB_METHOD, SAMPLING_FALLBACK_METHOD, LogprobGrader
from .scope import SET_DIFFERENCE_METHOD, ToolScopeGrader

__all__ = [
    "LOGPROB_METHOD",
    "SAMPLING_FALLBACK_METHOD",
    "SET_DIFFERENCE_METHOD",
    "LogprobGrader",
    "ToolScopeGrader",
]
