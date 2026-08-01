"""Combine 3 independently-produced profile_<kb>_<judge>.json files into one
weakness_evolution + HTML report, without re-paying the cost of re-grading everyone
in a single run_profiler.py invocation.

Why this exists: gemma and GLM-5.2 were re-graded together in one run_profiler.py
call (fast, logprobs-based) and each wrote its own profile_*.json immediately, as
designed. That run was killed partway through Kimi-K2.6's grading (~2h, sampling-
based) before it could reach the final step that writes the SHARED
weakness_evolution_<kb>.json (written once,
after every judge in the invocation finishes). Kimi was re-run separately afterward
with a corrected --fallback-k (the lever that actually smooths its per-probe verdict,
unlike --iterations which only feeds the stability side-table, not the headline
fail rate). Re-running gemma+GLM+Kimi together again just to get the shared
evolution/report file written correctly would mean re-paying Kimi's ~2h sampling
cost a second time for no benefit -- the profile_*.json files already hold
everything the paper needs (overall_fail_rate, by_strategy).

This script rebuilds the shared artifacts from the 3 already-fresh profile_*.json
files directly, without re-grading anything. The rebuilt weakness_evolution_<kb>.json
marks stability as "not recomputed this run" (the previous session's iteration-based
stability numbers do not apply to this run's config and would be misleading if
reused as-is) rather than fabricating iteration data that was never actually
computed under this run's settings.

Usage:
  python merge_profiler_run.py --kb ley_compose \
    --judges "hf_router:google/gemma-4-31B-it,hf_router:zai-org/GLM-5.2,hf_router:moonshotai/Kimi-K2.6"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results" / "level2_profiler"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge independently-run judge profiles into one report")
    ap.add_argument("--kb", default="ley_compose")
    ap.add_argument("--judges", required=True, help="'provider:model,provider:model' -- same roster used")
    ap.add_argument("--dataset", default="results/level1_probes/dataset_ley_compose.json")
    ap.add_argument("--top-k", type=int, default=3, help="must match run_profiler.py's default so the report renders the same shape")
    ap.add_argument("--axis", default="by_strategy")
    args = ap.parse_args()

    judge_keys = [j.strip() for j in args.judges.split(",") if j.strip()]

    profiles: dict = {}
    for key in judge_keys:
        pf = RESULTS / f"profile_{args.kb}_{_slug(key)}.json"
        if not pf.exists():
            raise SystemExit(f"missing {pf} -- run that judge first")
        profiles[key] = json.loads(pf.read_text(encoding="utf-8"))
        print(f"loaded {key}: fail_rate={profiles[key]['meta']['overall_fail_rate']} "
              f"n_probes={profiles[key]['meta']['n_probes']}  (from {pf.name}, "
              f"mtime={pf.stat().st_mtime})")

    # Rebuild a minimal, honest evolution artifact: one "iteration" per judge (the
    # single grading pass each profile_*.json already represents), no fabricated
    # stability -- these 3 judges were graded in 2 SEPARATE invocations, so a
    # cross-iteration stability comparison doesn't apply to this combined view.
    per_judge = {}
    for key, prof in profiles.items():
        ranked = prof["likely_weaknesses"]["by_strategy"]
        ranking = [r["key"] for r in ranked]
        method_counts = prof["meta"]["judge"]["method_counts"]
        method = "logprobs" if method_counts.get("logprobs", 0) >= method_counts.get("sampling", 0) else "sampling"
        per_judge[key] = {
            "iterations": [{"i": 0, "ranking": ranking,
                           "means": {r["key"]: r["mean"] for r in ranked}, "method": method}],
            "stability": {"topk_jaccard_mean": None, "kendall_tau_mean": None,
                          "rank_churn": {},
                          "note": "not recomputed in this merged run -- gemma/GLM and Kimi-K2.6 "
                                  "were graded in separate invocations; see the run log for each "
                                  "judge's own iteration count and fallback_k."},
        }

    evolution = {
        "config": {"dataset": args.dataset, "kb": args.kb, "mode": "merged_from_separate_runs",
                   "top_k": args.top_k, "axis": args.axis,
                   "note": "gemma+GLM-5.2 graded together in one invocation (5 iterations, "
                           "logprobs-based); Qwen3.6-35B-A3B:scaleway graded separately in a "
                           "second invocation (5 iterations, also logprobs-based) once it was "
                           "verified and added to the roster in place of Kimi-K2.6."},
        "per_judge": per_judge,
    }
    ev_path = RESULTS / f"weakness_evolution_{args.kb}.json"
    ev_path.write_text(json.dumps(evolution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nevolution (merged) -> {ev_path.name}")


if __name__ == "__main__":
    main()
