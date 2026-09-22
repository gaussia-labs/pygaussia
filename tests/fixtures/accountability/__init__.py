"""The accountability sandbox, as Datasets.

Eighteen hand-written sessions, each carrying the violations it contains and the figures the
metrics should return. They come from the paper's own sandbox
(Alquimia-ai/experiments, metrics/2026-09-accountability/sandbox/), so the expected blocks are an
oracle written before the SDK implementation existed rather than a recording of what it produces.

`expected.oversight_weighted` is derived from the hand-written `checks` of each planted violation,
not from the metric, so it exercises check detection and stratification independently. The
noisy-OR arithmetic itself is covered separately in `test_accountability.py`.
"""

import json
from functools import lru_cache
from pathlib import Path

from gaussia.schemas.common import Batch, Dataset

_SESSIONS = Path(__file__).parent / "sessions.json"


@lru_cache(maxsize=1)
def _payload() -> dict:
    return json.loads(_SESSIONS.read_text(encoding="utf-8"))


def sandbox_datasets() -> list[Dataset]:
    """Every sandbox session, in file order."""
    assistant_id = _payload()["assistant_id"]
    return [
        Dataset(
            session_id=session["session_id"],
            assistant_id=assistant_id,
            context=session["context"],
            oversight_policy=session["oversight_policy"],
            conversation=[
                Batch(
                    qa_id=turn["qa_id"],
                    query=turn["query"],
                    assistant=turn["assistant"],
                    ground_truth_assistant=turn.get("ground_truth_assistant", ""),
                    agentic=turn.get("agentic") or {},
                )
                for turn in session["conversation"]
            ],
        )
        for session in _payload()["sessions"]
    ]


def sandbox_expectations() -> dict[str, dict]:
    """The planted figures, keyed by session id."""
    return {session["session_id"]: session["expected"] for session in _payload()["sessions"]}


def sandbox_dataset(session_id: str) -> Dataset:
    """One session by id, for a test that only needs the case it is about."""
    for dataset in sandbox_datasets():
        if dataset.session_id == session_id:
            return dataset
    raise KeyError(session_id)
