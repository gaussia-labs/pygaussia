from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import AliasChoices, AliasPath, BaseModel, Field, ValidationError, model_validator

from gaussia import Batch, Dataset, Retriever

DEFAULT_DATASET_LANGUAGE = "english"


class RawMessage(BaseModel):
    type: str
    data: dict[str, Any]


class ConversationPayload(BaseModel):
    messages: list[RawMessage]
    blobs: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str
    user_id: str | None = None


class ContextPersistancePayload(BaseModel):
    stream_id: str
    control_id: str
    assistant_id: str
    assistant_context: str
    conversation: ConversationPayload = Field(validation_alias=AliasChoices("conversation", "chat_history"))
    agentspace_id: str | None = None
    collection_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] | None = None
    persistance_strategy: str | None = None
    depth: int = 0
    source: Literal["context.persistance.v1"] = "context.persistance.v1"
    session_id: str = Field(
        validation_alias=AliasChoices(
            "session_id",
            AliasPath("conversation", "session_id"),
            AliasPath("chat_history", "session_id"),
        )
    )

    @model_validator(mode="after")
    def validate_messages(self) -> ContextPersistancePayload:
        if not self.conversation.messages:
            raise ValueError("ContextPersistance conversation.messages must be a non-empty list")
        return self

    def to_canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True)
class BenchmarkInput:
    dataset: Dataset
    metadata: dict[str, str]
    payload_source: Literal["dataset", "context_persistance"]
    legacy_payload: ContextPersistancePayload | None = None


def load_benchmark_input(parameters: Mapping[str, Any]) -> BenchmarkInput:
    dataset_payload = parameters.get("dataset")
    if dataset_payload is not None:
        if not isinstance(dataset_payload, Mapping):
            raise ValueError("JobSpec.parameters.dataset must be a JSON object")
        dataset = _load_dataset(dataset_payload)
        metadata = _normalize_metadata(parameters.get("metadata") or {})
        _add_dataset_metadata_defaults(metadata, dataset)
        return BenchmarkInput(dataset=dataset, metadata=metadata, payload_source="dataset")

    context_payload = parameters.get("context_persistance")
    if not isinstance(context_payload, Mapping):
        raise ValueError("JobSpec.parameters must include dataset or context_persistance JSON object")  # noqa: TRY004

    payload = load_context_persistance_payload(context_payload)
    dataset = build_dataset_from_context_persistance(payload)
    return BenchmarkInput(
        dataset=dataset,
        metadata=_metadata_from_context_persistance(payload),
        payload_source="context_persistance",
        legacy_payload=payload,
    )


def load_context_persistance_payload(payload: Mapping[str, Any]) -> ContextPersistancePayload:
    try:
        return ContextPersistancePayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc)) from exc


def build_dataset_from_context_persistance(payload: ContextPersistancePayload) -> Dataset:
    conversation = build_batches_from_context_persistance(payload)
    if not conversation:
        raise ValueError(
            "No human/assistant interactions could be derived from context_persistance conversation.messages"
        )

    return Dataset(
        session_id=payload.session_id,
        assistant_id=payload.assistant_id,
        language=DEFAULT_DATASET_LANGUAGE,
        context=payload.assistant_context,
        conversation=conversation,
    )


def build_static_retriever(dataset: Dataset) -> type[Retriever]:
    class StaticRetriever(Retriever):
        def load_dataset(self) -> Sequence[Dataset]:
            return [dataset]

    return StaticRetriever


def build_batches_from_context_persistance(payload: ContextPersistancePayload) -> list[Batch]:
    batches: list[Batch] = []
    pending_query: str | None = None
    pair_index = 0

    for message in payload.conversation.messages:
        role = normalize_role(message.type)
        content = extract_content(message)

        if role == "human":
            pending_query = content
            continue

        if role == "ai" and pending_query:
            pair_index += 1
            batches.append(
                Batch(
                    qa_id=f"{payload.control_id}-{pair_index}",
                    query=pending_query,
                    assistant=content,
                    ground_truth_assistant="",
                    logprobs={},
                    observation=None,
                    agentic={},
                    ground_truth_agentic={},
                )
            )
            pending_query = None

    return batches


def normalize_role(message_type: str) -> str:
    lowered = message_type.lower()
    if lowered in {"human", "user"}:
        return "human"
    if lowered in {"ai", "assistant"}:
        return "ai"
    return lowered


def extract_content(message: RawMessage) -> str:
    content = message.data.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part)
    return str(content).strip()


def _load_dataset(payload: Mapping[str, Any]) -> Dataset:
    try:
        dataset = Dataset.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc, prefix="dataset")) from exc
    if not dataset.conversation:
        raise ValueError("JobSpec.parameters.dataset.conversation must be a non-empty list")
    return dataset


def _normalize_metadata(raw_metadata: Any) -> dict[str, str]:
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("JobSpec.parameters.metadata must be a JSON object when provided")

    metadata: dict[str, str] = {}
    for key, value in raw_metadata.items():
        if value is None:
            metadata[str(key)] = ""
        elif isinstance(value, str | int | float | bool):
            metadata[str(key)] = str(value)
        else:
            metadata[str(key)] = str(value)
    return metadata


def _add_dataset_metadata_defaults(metadata: dict[str, str], dataset: Dataset) -> None:
    metadata.setdefault("assistant_id", dataset.assistant_id)
    metadata.setdefault("session_id", dataset.session_id)
    metadata.setdefault("source", "gaussia.dataset.v1")


def _metadata_from_context_persistance(payload: ContextPersistancePayload) -> dict[str, str]:
    return {
        "assistant_id": payload.assistant_id,
        "session_id": payload.session_id,
        "stream_id": payload.stream_id,
        "control_id": payload.control_id,
        "agentspace_id": payload.agentspace_id or "",
        "source": payload.source,
    }


def _format_validation_error(exc: ValidationError, prefix: str | None = None) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", []))
        if prefix and location:
            location = f"{prefix}.{location}"
        elif prefix:
            location = prefix
        message = error.get("msg", "invalid value")
        parts.append(f"{location}: {message}")
    return "; ".join(parts)
