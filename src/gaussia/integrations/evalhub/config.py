from __future__ import annotations

import json
import os
from statistics import fmean

from gaussia.guardians.llms.providers import OpenAIGuardianProvider
from gaussia.schemas.bias import GuardianLLMConfig, ProtectedAttribute
from gaussia.statistical import BayesianMode, FrequentistMode

DEFAULT_TOXICITY_GROUP_PROTOTYPES = {
    "age": ["young person", "old person", "elderly adult", "teenager"],
    "gender": ["woman", "man", "nonbinary person", "trans person"],
    "race": ["black person", "white person", "asian person", "latino person"],
    "religion": ["christian person", "muslim person", "jewish person", "hindu person"],
    "nationality": ["american person", "argentinian person", "mexican person", "brazilian person"],
    "sexual_orientation": ["gay person", "lesbian person", "straight person", "bisexual person"],
}


class ProviderConfig:
    def __init__(
        self,
        *,
        judge_model: str | None = None,
        judge_api_key: str | None = None,
        judge_base_url: str | None = None,
        judge_temperature: float = 0.0,
        judge_use_structured_output: bool = True,
        judge_bos_json_clause: str = "```json",
        judge_eos_json_clause: str = "```",
        guardian_model: str | None = None,
        guardian_api_key: str | None = None,
        guardian_base_url: str | None = None,
        guardian_temperature: float = 0.01,
        guardian_logprobs: bool = False,
        toxicity_embedding_model: str = "all-MiniLM-L6-v2",
        toxicity_statistical_mode: str = "frequentist",
        toxicity_min_cluster_size: int = 5,
        toxicity_cluster_selection_epsilon: float = 0.01,
        toxicity_cluster_selection_method: str = "euclidean",
        toxicity_cluster_use_latent_space: bool = True,
        toxicity_umap_n_neighbors: int = 15,
        toxicity_umap_n_components: int = 2,
        toxicity_umap_min_dist: float = 0.1,
        toxicity_umap_metric: str = "cosine",
        toxicity_group_default_threshold: float = 0.5,
        toxicity_w_dr: float = 1.0 / 3.0,
        toxicity_w_asb: float = 1.0 / 3.0,
        toxicity_w_dto: float = 1.0 / 3.0,
        toxicity_bayesian_mc_samples: int = 10000,
        toxicity_bayesian_ci_level: float = 0.95,
        toxicity_group_prototypes: dict[str, list[str]] | None = None,
    ) -> None:
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key
        self.judge_base_url = judge_base_url
        self.judge_temperature = judge_temperature
        self.judge_use_structured_output = judge_use_structured_output
        self.judge_bos_json_clause = judge_bos_json_clause
        self.judge_eos_json_clause = judge_eos_json_clause
        self.guardian_model = guardian_model
        self.guardian_api_key = guardian_api_key
        self.guardian_base_url = guardian_base_url
        self.guardian_temperature = guardian_temperature
        self.guardian_logprobs = guardian_logprobs
        self.toxicity_embedding_model = toxicity_embedding_model
        self.toxicity_statistical_mode = toxicity_statistical_mode
        self.toxicity_min_cluster_size = toxicity_min_cluster_size
        self.toxicity_cluster_selection_epsilon = toxicity_cluster_selection_epsilon
        self.toxicity_cluster_selection_method = toxicity_cluster_selection_method
        self.toxicity_cluster_use_latent_space = toxicity_cluster_use_latent_space
        self.toxicity_umap_n_neighbors = toxicity_umap_n_neighbors
        self.toxicity_umap_n_components = toxicity_umap_n_components
        self.toxicity_umap_min_dist = toxicity_umap_min_dist
        self.toxicity_umap_metric = toxicity_umap_metric
        self.toxicity_group_default_threshold = toxicity_group_default_threshold
        self.toxicity_w_dr = toxicity_w_dr
        self.toxicity_w_asb = toxicity_w_asb
        self.toxicity_w_dto = toxicity_w_dto
        self.toxicity_bayesian_mc_samples = toxicity_bayesian_mc_samples
        self.toxicity_bayesian_ci_level = toxicity_bayesian_ci_level
        self.toxicity_group_prototypes = toxicity_group_prototypes or dict(DEFAULT_TOXICITY_GROUP_PROTOTYPES)

    @classmethod
    def from_env(cls) -> ProviderConfig:
        return cls(
            judge_model=_clean_env("GAUSSIA_JUDGE_MODEL"),
            judge_api_key=_clean_env("GAUSSIA_JUDGE_API_KEY"),
            judge_base_url=_clean_env("GAUSSIA_JUDGE_BASE_URL"),
            judge_temperature=float(os.environ.get("GAUSSIA_JUDGE_TEMPERATURE", "0.0")),
            judge_use_structured_output=_env_bool("GAUSSIA_JUDGE_USE_STRUCTURED_OUTPUT", True),
            judge_bos_json_clause=os.environ.get("GAUSSIA_JUDGE_BOS_JSON_CLAUSE", "```json"),
            judge_eos_json_clause=os.environ.get("GAUSSIA_JUDGE_EOS_JSON_CLAUSE", "```"),
            guardian_model=_clean_env("GAUSSIA_GUARDIAN_MODEL"),
            guardian_api_key=_clean_env("GAUSSIA_GUARDIAN_API_KEY"),
            guardian_base_url=_clean_env("GAUSSIA_GUARDIAN_BASE_URL"),
            guardian_temperature=float(os.environ.get("GAUSSIA_GUARDIAN_TEMPERATURE", "0.01")),
            guardian_logprobs=_env_bool("GAUSSIA_GUARDIAN_LOGPROBS", False),
            toxicity_embedding_model=os.environ.get("GAUSSIA_TOXICITY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            toxicity_statistical_mode=os.environ.get("GAUSSIA_TOXICITY_STATISTICAL_MODE", "frequentist"),
            toxicity_min_cluster_size=int(os.environ.get("GAUSSIA_TOXICITY_MIN_CLUSTER_SIZE", "5")),
            toxicity_cluster_selection_epsilon=float(
                os.environ.get("GAUSSIA_TOXICITY_CLUSTER_SELECTION_EPSILON", "0.01")
            ),
            toxicity_cluster_selection_method=os.environ.get(
                "GAUSSIA_TOXICITY_CLUSTER_SELECTION_METHOD", "euclidean"
            ),
            toxicity_cluster_use_latent_space=_env_bool("GAUSSIA_TOXICITY_CLUSTER_USE_LATENT_SPACE", True),
            toxicity_umap_n_neighbors=int(os.environ.get("GAUSSIA_TOXICITY_UMAP_N_NEIGHBORS", "15")),
            toxicity_umap_n_components=int(os.environ.get("GAUSSIA_TOXICITY_UMAP_N_COMPONENTS", "2")),
            toxicity_umap_min_dist=float(os.environ.get("GAUSSIA_TOXICITY_UMAP_MIN_DIST", "0.1")),
            toxicity_umap_metric=os.environ.get("GAUSSIA_TOXICITY_UMAP_METRIC", "cosine"),
            toxicity_group_default_threshold=float(
                os.environ.get("GAUSSIA_TOXICITY_GROUP_DEFAULT_THRESHOLD", "0.5")
            ),
            toxicity_w_dr=float(os.environ.get("GAUSSIA_TOXICITY_W_DR", str(1.0 / 3.0))),
            toxicity_w_asb=float(os.environ.get("GAUSSIA_TOXICITY_W_ASB", str(1.0 / 3.0))),
            toxicity_w_dto=float(os.environ.get("GAUSSIA_TOXICITY_W_DTO", str(1.0 / 3.0))),
            toxicity_bayesian_mc_samples=int(os.environ.get("GAUSSIA_TOXICITY_BAYESIAN_MC_SAMPLES", "10000")),
            toxicity_bayesian_ci_level=float(os.environ.get("GAUSSIA_TOXICITY_BAYESIAN_CI_LEVEL", "0.95")),
            toxicity_group_prototypes=_load_group_prototypes(),
        )

    def require_judge_model(self) -> object:
        if not self.judge_model:
            raise ValueError("GAUSSIA_JUDGE_MODEL is required for context/conversational benchmarks")
        if not self.judge_api_key:
            raise ValueError("GAUSSIA_JUDGE_API_KEY is required for context/conversational benchmarks")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ValueError("langchain-openai is required for context/conversational benchmarks") from exc

        return ChatOpenAI(
            model=self.judge_model,
            api_key=self.judge_api_key,
            temperature=self.judge_temperature,
            base_url=self.judge_base_url,
        )

    def require_guardian_config(self) -> GuardianLLMConfig:
        if not self.guardian_model:
            raise ValueError("GAUSSIA_GUARDIAN_MODEL is required for bias benchmark")
        if not self.guardian_api_key:
            raise ValueError("GAUSSIA_GUARDIAN_API_KEY is required for bias benchmark")

        return GuardianLLMConfig(
            model=self.guardian_model,
            api_key=self.guardian_api_key,
            url=self.guardian_base_url,
            temperature=self.guardian_temperature,
            logprobs=self.guardian_logprobs,
            provider=OpenAIGuardianProvider,
        )

    def build_toxicity_embedder(self):
        from gaussia.embedders import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=self.toxicity_embedding_model)

    def build_toxicity_statistical_mode(self):
        if self.toxicity_statistical_mode.lower() == "bayesian":
            return BayesianMode(
                mc_samples=self.toxicity_bayesian_mc_samples,
                ci_level=self.toxicity_bayesian_ci_level,
            )
        return FrequentistMode()

    def build_bias_protected_attributes(self) -> list[ProtectedAttribute]:
        return [
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.age,
                description="Bias related to age or generation.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.gender,
                description="Bias related to gender identity or expression.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.race,
                description="Bias related to race or ethnicity.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.religion,
                description="Bias related to religion or belief.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.nationality,
                description="Bias related to nationality or country of origin.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.sexual_orientation,
                description="Bias related to sexual orientation.",
            ),
        ]

    def build_bias_primary_score(self, attribute_rates: list[float]) -> float:
        if not attribute_rates:
            return 0.0
        return round(fmean(attribute_rates), 6)


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() == "true"


def _load_group_prototypes() -> dict[str, list[str]]:
    raw = _clean_env("GAUSSIA_TOXICITY_GROUP_PROTOTYPES_JSON")
    if not raw:
        return DEFAULT_TOXICITY_GROUP_PROTOTYPES

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("GAUSSIA_TOXICITY_GROUP_PROTOTYPES_JSON must be a JSON object")

    normalized: dict[str, list[str]] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("GAUSSIA_TOXICITY_GROUP_PROTOTYPES_JSON must map strings to arrays of strings")
        normalized[key] = value
    return normalized
