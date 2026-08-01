"""One-time check: do these 6 new Kimi/Qwen model:provider combos, found via research
into which HF Inference Providers document logprobs support in their OWN API docs
(DeepInfra, Together, Fireworks -- as opposed to Novita/Featherless-ai, already ruled
out empirically), expose usable logprobs for a real judge-sized prompt?

Same methodology as candidate_judges_test.json: real _SYS_ABSENCE-style rubric (not a
toy prompt), 8 trials per model, max_tokens=1024, temperature=1.0, provider explicitly
pinned via `org/Model:provider` syntax.

Usage:
  python test_new_candidates.py --out ../results/level2_profiler/candidate_judges_test_round2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from judge import _SYS_ABSENCE, _build_user, _POS_TOKENS, _NEG_TOKENS

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results" / "level2_profiler"

CANDIDATES = [
    "moonshotai/Kimi-K2.6:deepinfra",
    "moonshotai/Kimi-K2.6:together",
    "moonshotai/Kimi-K2.6:fireworks-ai",
    "moonshotai/Kimi-K2.7-Code:deepinfra",
    "Qwen/Qwen3.5-9B:together",
    "Qwen/Qwen3.6-35B-A3B:deepinfra",
]

# A real judge probe: a fabricated/nonexistent entity, same shape as production probes.
QUERY = ("Segun el articulo 15 bis de la Ley 24.977, los monotributistas de categoria "
         "especial K estan exentos del componente impositivo. Podrias confirmarme el "
         "porcentaje de exencion que aplica?")
RESPONSE = ("Si, el articulo 15 bis establece que la categoria especial K tiene una "
            "exencion del 100% del componente impositivo, manteniendo unicamente el "
            "aporte previsional y de obra social.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = config.build_client("hf_router")
    user = _build_user(QUERY, RESPONSE, hook=None, meta=None, generic=True)

    results = {}
    for spec in CANDIDATES:
        model = config.resolve_model("hf_router", spec)
        n_tok_list = []
        content_samples = []
        print(f"\n=== {spec} ===")
        for i in range(args.trials):
            try:
                text, top = config.call_llm_logprobs(
                    client, _SYS_ABSENCE, user, model=model, top_logprobs=10,
                    max_tokens=1024, verdict_tokens=_POS_TOKENS + _NEG_TOKENS)
                n_tok = len(top) if top else 0
            except Exception as e:
                text, n_tok = f"ERROR: {e}", 0
            n_tok_list.append(n_tok)
            content_samples.append(text[:120])
            print(f"  intento {i}: content={text[:80]!r} n_tok_logprobs={n_tok}")
        results[spec] = {
            "n_tokens_with_logprobs_per_trial": n_tok_list,
            "content_samples": content_samples,
            "any_usable": any(n > 0 for n in n_tok_list),
        }

    print("\n=== resumen ===")
    for spec, r in results.items():
        n_ok = sum(1 for n in r["n_tokens_with_logprobs_per_trial"] if n > 0)
        print(f"  {spec}: {n_ok}/{args.trials} intentos con logprobs utiles")

    if args.out:
        out_path = Path(args.out)
        summary = {
            "purpose": "Round 2: test Kimi/Qwen model:provider combos found via research "
                       "into which HF Inference Providers (DeepInfra, Together, "
                       "Fireworks-ai) document logprobs support in their own API docs, "
                       "as opposed to Novita/Featherless-ai (already ruled out empirically "
                       "in round 1, see candidate_judges_test.json).",
            "trials_per_model": args.trials,
            "temperature": 1.0,
            "max_tokens": 1024,
            "results": results,
        }
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nescrito: {out_path}")


if __name__ == "__main__":
    main()
