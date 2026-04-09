"""Common data models for Gaussia."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Logprobs(BaseModel):
    """
    Logprobs are the log probabilities of the tokens in the response.
    """

    tokens: list[str]
    token_logprobs: list[float]


class Batch(BaseModel):
    """
    A batch represents a single interaction between a user and an assistant in a conversation.
    Each batch contains the user's query, the assistant's response, and optional metadata.

    Attributes:
        qa_id (str): Unique identifier for the batch
        ground_truth_assistant (str): The expected or reference response from the assistant
        observation (Optional[str]): Any additional observations or notes about the interaction
        assistant (str): The actual response provided by the assistant
        query (str): The user's input or question
        agentic (Optional[dict]): Additional metadata or context about the agent's behavior
        ground_truth_agentic (Optional[dict]): Expected or reference metadata for the agent's behavior
        logprobs (Optional[dict]): Log probabilities for tokens
    """

    ground_truth_assistant: str
    logprobs: dict | None = {}
    observation: str | None = None
    assistant: str
    query: str
    agentic: dict | None = {}
    ground_truth_agentic: dict | None = {}
    qa_id: str
    weight: float | None = Field(default=None, ge=0)


class IterationLevel(StrEnum):
    FULL_DATASET = "full_dataset"
    STREAM_SESSIONS = "stream_sessions"
    STREAM_BATCHES = "stream_batches"


class SessionMetadata(BaseModel):
    session_id: str
    assistant_id: str
    language: str | None = "english"
    context: str


class StreamedBatch(BaseModel):
    metadata: SessionMetadata
    batch: "Batch"


class Dataset(BaseModel):
    """
    A dataset represents a complete conversation session between a user and an assistant.
    It contains metadata about the session and a list of all interactions (batches).

    Attributes:
        session_id (str): Unique identifier for the conversation session
        assistant_id (str): Identifier for the specific assistant involved
        language (str): The language used in the conversation
        context (str): Additional context or background information for the conversation
        conversation (list[Batch]): List of all interactions (batches) in the conversation
    """

    session_id: str
    assistant_id: str
    language: str | None = "english"
    context: str
    conversation: list[Batch]
