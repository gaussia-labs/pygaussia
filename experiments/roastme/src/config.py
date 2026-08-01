"""Experiment config: LLM provider registry + .env loading.

The LLM is CONFIGURABLE (provider + model) so several can be compared. Everything goes
through the OpenAI-compatible pattern (same as prompt-leakage/experiments): Groq and any
compatible endpoint share one client. The model is resolved from a flag, the environment,
or the provider default.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
load_dotenv(HERE / ".env")


# Registry of OpenAI-compatible providers. Adding one is a single entry here.
#   groq/openai : cloud API, model chosen by id on each call.
#   hf          : DEDICATED HuggingFace Inference Endpoint (one model per endpoint URL).
#                 The endpoint already pins the model; set it to "scale to zero" when idle.
#                 base_url and model come from the environment.
#   supports_logprobs: whether the provider exposes logprobs on /chat/completions. A HINT,
#     not a guarantee: the judge still falls back to sampling if the response omits them.
PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        # Verified empirically: Groq /chat/completions answers 400 "logprobs is not
        # supported", so the judge goes straight to sampling and skips the wasted call.
        "supports_logprobs": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "supports_logprobs": True,
    },
    "hf": {
        "base_url_env": "HF_ENDPOINT",    # dedicated endpoint URL (pins the model)
        "base_suffix": "/v1",             # HF endpoints expose an OpenAI-compatible /v1
        "key_env": "HF_TOKEN",
        "default_model_env": "HF_MODEL",
        "supports_logprobs": True,        # TGI exposes logprobs; confirmed by probe_logprobs_support()
    },
    # HF Inference Providers (router): SERVERLESS, pay per token. HF routes to a provider that
    # already hosts the model (Together/Fireworks/Novita/...). No endpoints to create and no
    # org write access needed, which is what makes the very large models reachable (GLM-5.2
    # 753B, Kimi-K2.6 1T). Model chosen by id on each call, like Groq.
    "hf_router": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "default_model": "google/gemma-4-31B-it",
        "bill_to_env": "HF_BILL_TO",   # bills usage to an org (X-HF-Bill-To header)
        # The router delegates to a downstream provider, so logprobs support varies per
        # provider. Assumed available; the judge falls back to sampling when they do not
        # arrive (typical for reasoners like GLM/Kimi).
        "supports_logprobs": True,
    },
}


def supports_logprobs(provider: str) -> bool:
    """Hint on whether the provider exposes logprobs; the judge still checks at runtime."""
    return bool(PROVIDERS.get(provider, {}).get("supports_logprobs", False))


def build_client(provider: str = "groq", *, base_url: str | None = None) -> OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider!r}. Options: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise RuntimeError(f"{cfg['key_env']} is missing from the environment/.env for provider={provider}")
    # Explicit base_url > provider env (hf) > fixed base_url (groq/openai).
    if base_url is None:
        if "base_url_env" in cfg:
            raw = os.environ.get(cfg["base_url_env"])
            if not raw:
                raise RuntimeError(f"{cfg['base_url_env']} is missing from the environment/.env for provider={provider}")
            base_url = raw.rstrip("/") + cfg.get("base_suffix", "")
        else:
            base_url = cfg["base_url"]
    # Optional header to bill usage to an org (HF Inference Providers: X-HF-Bill-To).
    default_headers = None
    bill_to = os.environ.get(cfg["bill_to_env"]) if "bill_to_env" in cfg else None
    if bill_to:
        default_headers = {"X-HF-Bill-To": bill_to}
    return OpenAI(base_url=base_url, api_key=key, timeout=120.0, max_retries=4,
                  default_headers=default_headers)


def resolve_model(provider: str = "groq", model: str | None = None) -> str:
    """Explicit model > provider env > default. For hf the model usually comes in HF_MODEL,
    or whatever the running endpoint pins."""
    if model:
        return model
    cfg = PROVIDERS.get(provider, {})
    if "default_model_env" in cfg:
        return os.environ.get(cfg["default_model_env"]) or ""
    env_key = f"{provider.upper()}_MODEL"
    return os.environ.get(env_key) or cfg.get("default_model", "")


def call_llm(client: OpenAI, system: str, user: str, *, model: str,
             temperature: float = 0.3, max_tokens: int = 1024,
             seed: int | None = None) -> str:
    """A single turn against an OpenAI-compatible endpoint.

    Careful: a low max_tokens truncates long answers (e.g. JSON for large tables).
    `seed` (where the provider honours it, e.g. Groq) reduces run-to-run variation but
    does not remove it, which is why the notebooks also load a frozen canonical dataset.
    """
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def call_llm_logprobs(client: OpenAI, system: str, user: str, *, model: str,
                      top_logprobs: int = 10, temperature: float = 1.0,
                      max_tokens: int = 1, verdict_tokens: tuple[str, ...] | None = None
                      ) -> tuple[str, list[dict] | None]:
    """One turn that requests token logprobs, for the logprob-based judge.

    Returns (text, top_logprobs_at_the_verdict_token). The second item is a list
    of {"token": str, "logprob": float}, or None when no usable logprobs were
    found (caller must fall back to sampling).

    Without `verdict_tokens`: reads the FIRST generated token's logprobs (correct
    for a model that answers immediately, e.g. a non-reasoning model with
    max_tokens=1 -- its only token IS the verdict).

    With `verdict_tokens` (surface forms to look for, e.g. SI/NO variants):
    reasoning models emit a chain-of-thought preamble before the verdict, so the
    FIRST token is never it -- this scans the full per-token logprobs sequence
    from the end backwards for the last token matching one of `verdict_tokens`
    and reads logprobs there instead. Give `max_tokens` enough room for the model
    to finish reasoning and reach that token (empirically ~500+ for GLM-5.2).
    Some providers/models (confirmed for Kimi-K2.6 via hf_router) never attach
    logprobs to the visible answer at all when the model reasons through a
    separate channel -- for those this returns None like any other unusable case,
    and the caller falls back to sampling.

    temperature=1.0 and top_logprobs=10 mirror the judge spec: at temp=1 the
    token distribution reflects the model's real uncertainty.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return text, None

    if verdict_tokens is None:
        entry = content[0]
    else:
        targets = {t.strip() for t in verdict_tokens}
        entry = next((c for c in reversed(content) if c.token.strip() in targets), None)
        if entry is None:
            return text, None

    top = getattr(entry, "top_logprobs", None)
    if not top:
        return text, None
    return text, [{"token": t.token, "logprob": t.logprob} for t in top]


# --- Target (asistente objetivo: runtime de Alquimia en Railway) ------------
def target_config() -> dict:
    """Read the config of the assistant under test. Secrets live in .env, never tracked."""
    base = os.environ.get("TARGET_BASE_URL", "").rstrip("/")
    token = os.environ.get("TARGET_API_TOKEN", "")
    assistant = os.environ.get("TARGET_ASSISTANT_ID", "")
    missing = [k for k, v in {"TARGET_BASE_URL": base, "TARGET_API_TOKEN": token,
                              "TARGET_ASSISTANT_ID": assistant}.items() if not v]
    if missing:
        raise RuntimeError("Faltan variables del target en .env: " + ", ".join(missing))
    return {"base_url": base, "token": token, "assistant_id": assistant}


# --- compatibilidad con el código portado del sandbox ----------------------
def build_groq_client() -> OpenAI:
    return build_client("groq")


def groq_model() -> str:
    return resolve_model("groq", os.environ.get("GROQ_MODEL"))


def call_groq(client: OpenAI, system: str, user: str, temperature: float = 0.3,
              max_tokens: int = 512) -> str:
    return call_llm(client, system, user, model=groq_model(),
                    temperature=temperature, max_tokens=max_tokens)
