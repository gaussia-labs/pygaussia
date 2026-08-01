"""One-time verification script: does the GLM-5.2 logprobs fix pick the right token?

judge.py's reasoner branch scans the FULL per-token logprobs sequence from the end
for a SI/NO surface form, instead of assuming the first token is the verdict (which
for a reasoning model is its chain-of-thought preamble, never the answer). Risk: a
long Spanish reasoning trace can legitimately contain the literal token "si"
(conditional "if", not affirmative "sí") or "no" many times before the real verdict,
so "last match from the end" could in principle grab an incidental grammatical token.

This script checks that empirically, on REAL frozen probes (not a toy example): for
each probe, it gets BOTH the model's own clean final answer text (`message.content`,
already separated from reasoning by the API) AND the logprobs-derived verdict from
the SAME call, and asserts they agree. If they don't, that's a real bug to fix before
trusting the pipeline -- not something to average away or ignore.

Usage:
  python verify_glm_logprobs.py --n 25
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config
import probe_library as pl
from judge import (_RUBRICS, _SYS_ABSENCE, _SYS_FALSE_PREMISE, _build_user,
                   _p_violation_from_logprobs, parse_leading_verdict,
                   _POS_TOKENS, _NEG_TOKENS)

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results" / "level2_profiler"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify GLM-5.2 logprobs verdict extraction")
    ap.add_argument("--n", type=int, default=25, help="probes to sample")
    ap.add_argument("--model", default="zai-org/GLM-5.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="reasoning budget before the verdict token; some reasoners "
                         "(e.g. Qwen3.5-9B) need much more than GLM-5.2's 1024")
    ap.add_argument("--out", default=None, help="write a JSON summary here (for the notebook)")
    args = ap.parse_args()

    dataset = HERE / "results" / "level1_probes" / "dataset_ley_compose.json"
    probes = pl.load_dataset(dataset)
    transcripts = json.loads((RESULTS / "transcripts_ley_compose.json").read_text(encoding="utf-8"))

    scoreable = [p for p in probes if p.plugin is not None and transcripts.get(p.id, {}).get("response")]
    random.seed(args.seed)
    sample = random.sample(scoreable, min(args.n, len(scoreable)))

    client = config.build_client("hf_router")
    model = config.resolve_model("hf_router", args.model)

    agree, mismatches, empty_content = 0, [], []
    no_logprobs = []
    for p in sample:
        resp = transcripts[p.id]["response"]
        rubric = _RUBRICS.get(p.plugin or "")
        if rubric is not None:
            system = rubric
            user = _build_user(p.query, resp, p.hook, p.meta, generic=True)
        else:
            system = _SYS_ABSENCE if p.hook.doc == 0 else _SYS_FALSE_PREMISE
            user = _build_user(p.query, resp, p.hook, p.meta)

        text, top = config.call_llm_logprobs(
            client, system, user, model=model, top_logprobs=10, max_tokens=args.max_tokens,
            verdict_tokens=_POS_TOKENS + _NEG_TOKENS)

        text_verdict = parse_leading_verdict(text)
        if top is None:
            no_logprobs.append(p.id)
            print(f"[{p.id}] NO LOGPROBS (text verdict={text_verdict}, content={text!r:.80})")
            continue

        p_violation = _p_violation_from_logprobs(top)
        logprobs_verdict = int(p_violation >= 0.5)

        if text_verdict is None:
            # empty/unparseable content: a stochastic hiccup (model didn't finish
            # cleanly at temperature=1), NOT evidence the token-search algorithm
            # picked the wrong token -- there was no clean final answer to compare
            # against at all. judge.py's safety net treats this case as "no logprobs"
            # and falls back to sampling; tracked separately here, not as a mismatch.
            print(f"[{p.id}] empty/unparseable content={text!r:.100} (safety net would skip this)")
            empty_content.append({"probe_id": p.id, "content": text})
            continue

        if logprobs_verdict == text_verdict:
            agree += 1
        else:
            mismatches.append({"probe_id": p.id, "content": text, "text_verdict": text_verdict,
                               "logprobs_verdict": logprobs_verdict, "p_violation": p_violation})
            print(f"[{p.id}] MISMATCH: text={text!r} (parsed={text_verdict}) "
                  f"vs logprobs_verdict={logprobs_verdict} (p={p_violation:.3f})")
            print(f"    top_logprobs at picked position: {top[:5]}")

    n_comparable = agree + len(mismatches)
    agreement_rate = agree / n_comparable if n_comparable else None
    print(f"\n=== {model} ===")
    print(f"sample={len(sample)}  no_logprobs={len(no_logprobs)}  "
          f"empty_content={len(empty_content)}  comparable={n_comparable}  "
          f"agree={agree}/{n_comparable}"
          + (f" ({agreement_rate:.1%})" if n_comparable else "  (no comparable cases)"))
    if mismatches:
        print(f"\n{len(mismatches)} REAL MISMATCHES -- investigate before trusting this path:")
        for m in mismatches:
            print(f"  {m['probe_id']}: content={m['content']!r} "
                  f"text_verdict={m['text_verdict']} logprobs_verdict={m['logprobs_verdict']}")

    if args.out:
        summary = {
            "model": model, "sample_size": len(sample), "seed": args.seed,
            "no_logprobs": len(no_logprobs), "empty_content": len(empty_content),
            "comparable": n_comparable, "agree": agree, "agreement_rate": agreement_rate,
            "mismatches": mismatches,
            "empty_content_probe_ids": [e["probe_id"] for e in empty_content],
        }
        out_path = Path(args.out)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nescrito: {out_path}")


if __name__ == "__main__":
    main()
