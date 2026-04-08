"""Generator schemas and interfaces for Gaussia.

This module defines the base classes and Pydantic models for
synthetic dataset generation from context documents.

The BaseGenerator class uses LangChain's BaseChatModel interface,
allowing any LangChain-compatible model to be used for generation.
"""

import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from .common import Batch, Dataset


class Chunk(BaseModel):
    """Represents a chunk of context for query generation.

    Attributes:
        content: The text content of the chunk
        chunk_id: Unique identifier for this chunk
        metadata: Optional metadata (e.g., header hierarchy, source file)
    """

    content: str
    chunk_id: str
    metadata: dict | None = Field(default_factory=dict)


class GeneratedQuery(BaseModel):
    """A single generated query from the LLM.

    Attributes:
        query: The generated question/query text
        difficulty: Optional difficulty level (easy, medium, hard)
        query_type: Type of query (factual, inferential, application, etc.)
    """

    query: str
    difficulty: str | None = None
    query_type: str | None = None


class GeneratedQueriesOutput(BaseModel):
    """Structured output schema for LLM query generation.

    Attributes:
        queries: List of generated queries for a chunk
        chunk_summary: Brief summary of the chunk content
    """

    queries: list[GeneratedQuery] = Field(description="List of generated queries based on the chunk content")
    chunk_summary: str = Field(description="Brief summary of the chunk content (1-2 sentences)")


class ConversationTurn(BaseModel):
    """A single turn in a contextualized conversation.

    Represents one query in a multi-turn conversation where each turn
    builds on the previous one for coherent dialogue flow.

    Attributes:
        query: The question/query for this turn
        expected_context: What this turn builds upon (context from previous turns)
        difficulty: Optional difficulty level (easy, medium, hard)
        query_type: Type of query (factual, inferential, follow-up, etc.)
        turn_number: Position in the conversation (1-indexed)
    """

    query: str
    expected_context: str | None = Field(
        default=None,
        description="Context this turn builds upon from previous turns",
    )
    difficulty: str | None = None
    query_type: str | None = None
    turn_number: int = Field(ge=1, description="Position in conversation (1-indexed)")


class GeneratedConversationOutput(BaseModel):
    """Structured output schema for conversation generation.

    Used when generating coherent multi-turn conversations from a chunk.

    Attributes:
        turns: List of conversation turns in order
        conversation_summary: Brief description of the conversation flow
        chunk_summary: Summary of the source chunk content
    """

    turns: list[ConversationTurn] = Field(
        description="Ordered list of conversation turns",
    )
    conversation_summary: str = Field(
        description="Brief summary of the conversation flow and topics covered",
    )
    chunk_summary: str = Field(
        description="Brief summary of the chunk content (1-2 sentences)",
    )


class BaseContextLoader(ABC):
    """Abstract base class for loading and chunking context documents.

    Context loaders are responsible for reading source documents and
    splitting them into chunks suitable for query generation.
    """

    def __init__(self, **kwargs):
        """Initialize the context loader with optional configuration.

        Args:
            **kwargs: Implementation-specific configuration parameters.
        """
        self.kwargs = kwargs

    @abstractmethod
    def load(self, source: str) -> list[Chunk]:
        """Load and chunk a context document.

        Args:
            source: Path to the document or document identifier.

        Returns:
            list[Chunk]: List of document chunks ready for processing.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")


class BaseChunkSelectionStrategy(ABC):
    """Abstract base class for chunk selection strategies.

    Strategies determine how chunks are selected and grouped for
    dataset generation. Each yielded group becomes a separate dataset.

    Example strategies:
        - Sequential: Process all chunks in order (single group)
        - Random Sampling: Randomly select k chunks, repeat n times
        - Clustering: Group related chunks together
    """

    @abstractmethod
    def select(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
        """Select and group chunks for processing.

        Args:
            chunks: All available chunks from the context loader.

        Yields:
            list[Chunk]: Groups of chunks to process together.
                Each group becomes a separate dataset.

        Example:
            >>> strategy = SequentialStrategy()
            >>> for chunk_group in strategy.select(chunks):
            ...     # Process this group as one dataset
            ...     pass
        """
        raise NotImplementedError("Must be implemented by subclass")


# Default prompts for query and conversation generation
DEFAULT_SYSTEM_PROMPT = """You are an expert at creating test questions for AI evaluation.
Your task is to generate diverse, high-quality questions based on the provided context.

Guidelines:
1. Generate questions that test different cognitive levels (recall, comprehension, application)
2. Ensure questions are grounded in the provided context
3. Vary question types: factual, inferential, comparative, analytical
4. Questions should be clear and unambiguous
5. Avoid yes/no questions; prefer open-ended questions that require substantive answers

Context:
{context}

{seed_examples_section}

Generate exactly {num_queries} questions based on this context."""


DEFAULT_CONVERSATION_PROMPT = """You are an expert at creating realistic test conversations for AI evaluation.
Your task is to generate a coherent multi-turn conversation based on the provided context.

Guidelines:
1. Start with a broad question about the main topic
2. Each subsequent turn should naturally follow from the previous one
3. Include follow-up questions that reference or clarify previous answers
4. Progress from simple recall to deeper analysis as the conversation develops
5. The conversation should feel natural, like a real user exploring a topic
6. Vary cognitive levels: start with recall, move to comprehension, then application/analysis

Context:
{context}

{seed_examples_section}

Generate a coherent conversation with exactly {num_turns} turns. Each turn should logically build on the previous ones."""


class BaseGenerator:
    """Base class for synthetic dataset generators using LangChain.

    This generator uses any LangChain-compatible chat model (BaseChatModel)
    to generate synthetic test queries from context documents.

    Args:
        model: LangChain BaseChatModel instance (e.g., ChatOpenAI, ChatGroq)
        use_structured_output: If True, use with_structured_output() for parsing

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini")
        generator = BaseGenerator(model=model)

        datasets = await generator.generate_dataset(
            context_loader=loader,
            source="./docs/knowledge_base.md",
            assistant_id="my-assistant",
        )
        ```
    """

    def __init__(
        self,
        model: BaseChatModel,
        use_structured_output: bool = True,
        **kwargs,
    ):
        """Initialize the generator with a LangChain model.

        Args:
            model: LangChain BaseChatModel instance
            use_structured_output: If True, use with_structured_output() for parsing
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.use_structured_output = use_structured_output
        self.kwargs = kwargs

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
    ) -> GeneratedQueriesOutput:
        """Call the LLM and parse response for queries.

        Args:
            prompt: User prompt
            system_prompt: System prompt for the LLM

        Returns:
            GeneratedQueriesOutput: Parsed structured output
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", prompt),
            ]
        )

        if self.use_structured_output:
            structured_model = self.model.with_structured_output(GeneratedQueriesOutput)
            chain = chat_prompt | structured_model
            return chain.invoke({})
        chain = chat_prompt | self.model
        response = chain.invoke({})
        content = str(response.content)
        return self._parse_json_response(content)

    async def _call_llm_conversation(
        self,
        prompt: str,
        system_prompt: str,
    ) -> GeneratedConversationOutput:
        """Call the LLM and parse response for conversations.

        Args:
            prompt: User prompt
            system_prompt: System prompt for the LLM

        Returns:
            GeneratedConversationOutput: Parsed structured output
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", prompt),
            ]
        )

        if self.use_structured_output:
            structured_model = self.model.with_structured_output(GeneratedConversationOutput)
            chain = chat_prompt | structured_model
            return chain.invoke({})
        chain = chat_prompt | self.model
        response = chain.invoke({})
        content = str(response.content)
        return self._parse_conversation_response(content)

    def _parse_json_response(self, content: str) -> GeneratedQueriesOutput:
        """Parse JSON from model response.

        Args:
            content: Raw model response content

        Returns:
            GeneratedQueriesOutput: Parsed output
        """
        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"No JSON found in response: {content[:200]}")

        return GeneratedQueriesOutput.model_validate_json(json_str)

    def _parse_conversation_response(self, content: str) -> GeneratedConversationOutput:
        """Parse JSON from model response for conversations.

        Args:
            content: Raw model response content

        Returns:
            GeneratedConversationOutput: Parsed output
        """
        # Try to extract JSON from markdown code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"No JSON found in response: {content[:200]}")

        return GeneratedConversationOutput.model_validate_json(json_str)

    async def generate_queries(
        self,
        chunk: Chunk,
        num_queries: int = 3,
        seed_examples: list[str] | None = None,
        custom_system_prompt: str | None = None,
    ) -> list[GeneratedQuery]:
        """Generate independent queries for a single chunk using LangChain.

        Args:
            chunk: Context chunk to generate queries from
            num_queries: Number of queries to generate
            seed_examples: Optional example queries for style guidance
            custom_system_prompt: Optional custom system prompt

        Returns:
            list[GeneratedQuery]: Generated queries
        """
        logger.debug(f"Generating {num_queries} queries for chunk {chunk.chunk_id}")

        seed_section = ""
        if seed_examples:
            examples_str = "\n".join(f"- {ex}" for ex in seed_examples)
            seed_section = f"\nExample questions (match this style):\n{examples_str}\n"

        system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        formatted_prompt = system_prompt.format(
            context=chunk.content,
            seed_examples_section=seed_section,
            num_queries=num_queries,
        )

        # Add JSON format instruction if not using structured output
        if not self.use_structured_output:
            formatted_prompt += """

You MUST respond with a valid JSON object in this exact format:
{{
    "queries": [
        {{"query": "Your question here", "difficulty": "easy|medium|hard", "query_type": "factual|inferential|comparative|analytical"}}
    ],
    "chunk_summary": "Brief 1-2 sentence summary of the context"
}}"""

        try:
            output = await self._call_llm(
                prompt=f"Generate {num_queries} questions.",
                system_prompt=formatted_prompt,
            )
            logger.debug(f"Generated {len(output.queries)} queries for chunk {chunk.chunk_id}")
            return output.queries
        except Exception as e:
            logger.error(f"Failed to generate queries for chunk {chunk.chunk_id}: {e}")
            raise

    async def generate_conversation(
        self,
        chunk: Chunk,
        num_turns: int = 3,
        seed_examples: list[str] | None = None,
        custom_system_prompt: str | None = None,
    ) -> list[ConversationTurn]:
        """Generate a coherent multi-turn conversation from a chunk.

        Creates follow-up questions where each turn builds on the previous,
        maintaining a natural conversation flow exploring the chunk's content.

        Args:
            chunk: The context chunk to generate conversation from.
            num_turns: Number of conversation turns to generate.
            seed_examples: Optional example conversations to guide style.
            custom_system_prompt: Optional custom system prompt.

        Returns:
            list[ConversationTurn]: Ordered conversation turns.
        """
        logger.debug(f"Generating {num_turns}-turn conversation for chunk {chunk.chunk_id}")

        seed_section = ""
        if seed_examples:
            examples_str = "\n".join(f"- {ex}" for ex in seed_examples)
            seed_section = f"\nExample conversation flow (match this style):\n{examples_str}\n"

        system_prompt = custom_system_prompt or DEFAULT_CONVERSATION_PROMPT
        formatted_prompt = system_prompt.format(
            context=chunk.content,
            seed_examples_section=seed_section,
            num_turns=num_turns,
        )

        # Add JSON format instruction if not using structured output
        if not self.use_structured_output:
            formatted_prompt += """

You MUST respond with a valid JSON object in this exact format:
{{
    "turns": [
        {{"query": "First question about the topic", "turn_number": 1, "difficulty": "easy", "query_type": "factual", "expected_context": null}},
        {{"query": "Follow-up question building on first", "turn_number": 2, "difficulty": "medium", "query_type": "inferential", "expected_context": "References the answer to turn 1"}},
        {{"query": "Deeper question exploring implications", "turn_number": 3, "difficulty": "hard", "query_type": "analytical", "expected_context": "Builds on turns 1 and 2"}}
    ],
    "conversation_summary": "Brief description of the conversation flow",
    "chunk_summary": "Brief 1-2 sentence summary of the context"
}}"""

        try:
            output = await self._call_llm_conversation(
                prompt=f"Generate a {num_turns}-turn conversation.",
                system_prompt=formatted_prompt,
            )
            logger.debug(f"Generated {len(output.turns)} turns for chunk {chunk.chunk_id}")
            return output.turns
        except Exception as e:
            logger.error(f"Failed to generate conversation for chunk {chunk.chunk_id}: {e}")
            raise

    async def generate_dataset(
        self,
        context_loader: BaseContextLoader,
        source: str,
        assistant_id: str,
        num_queries_per_chunk: int = 3,
        language: str = "english",
        seed_examples: list[str] | None = None,
        custom_system_prompt: str | None = None,
        selection_strategy: Optional["BaseChunkSelectionStrategy"] = None,
        conversation_mode: bool = False,
    ) -> list[Dataset]:
        """Generate datasets from a context source.

        Args:
            context_loader: Loader to chunk the source document
            source: Path to the context document
            assistant_id: ID for the assistant being tested
            num_queries_per_chunk: Queries/turns per chunk
            language: Language for queries
            seed_examples: Example queries for style guidance
            custom_system_prompt: Override default system prompt
            selection_strategy: Strategy for selecting/grouping chunks.
                Defaults to SequentialStrategy (process all chunks once).
            conversation_mode: If True, generate coherent multi-turn
                conversations instead of independent queries.

        Returns:
            list[Dataset]: Generated datasets (one per chunk group from strategy).
        """
        # Import here to avoid circular imports
        from pygaussia.generators.strategies import SequentialStrategy

        logger.info(f"Loading context from: {source}")
        chunks = context_loader.load(source)
        logger.info(f"Loaded {len(chunks)} chunks from source")

        # Use default sequential strategy if none provided
        strategy = selection_strategy or SequentialStrategy()
        logger.info(f"Using chunk selection strategy: {strategy}")

        datasets: list[Dataset] = []

        # Process each chunk group from the strategy
        for chunk_group in strategy.select(chunks):
            session_id = str(uuid.uuid4())
            batches: list[Batch] = []
            full_context = "\n\n".join(chunk.content for chunk in chunk_group)

            if conversation_mode:
                # Generate coherent conversations per chunk
                for chunk in chunk_group:
                    turns = await self.generate_conversation(
                        chunk=chunk,
                        num_turns=num_queries_per_chunk,
                        seed_examples=seed_examples,
                        custom_system_prompt=custom_system_prompt,
                    )

                    for turn in turns:
                        qa_id = f"{chunk.chunk_id}_t{turn.turn_number}"
                        batch = Batch(
                            qa_id=qa_id,
                            query=turn.query,
                            assistant="",
                            ground_truth_assistant="",
                            observation=f"Conversation turn {turn.turn_number} from chunk: {chunk.chunk_id}",
                            agentic=_build_conversation_metadata(chunk, turn),
                            ground_truth_agentic={},
                        )
                        batches.append(batch)
            else:
                # Generate independent queries per chunk
                for chunk in chunk_group:
                    queries = await self.generate_queries(
                        chunk=chunk,
                        num_queries=num_queries_per_chunk,
                        seed_examples=seed_examples,
                        custom_system_prompt=custom_system_prompt,
                    )

                    for i, generated_query in enumerate(queries):
                        qa_id = f"{chunk.chunk_id}_q{i + 1}"
                        batch = Batch(
                            qa_id=qa_id,
                            query=generated_query.query,
                            assistant="",
                            ground_truth_assistant="",
                            observation=f"Generated from chunk: {chunk.chunk_id}",
                            agentic=_build_agentic_metadata(chunk, generated_query),
                            ground_truth_agentic={},
                        )
                        batches.append(batch)

            logger.info(f"Generated {len(batches)} batches for chunk group " f"({len(chunk_group)} chunks)")

            dataset = Dataset(
                session_id=session_id,
                assistant_id=assistant_id,
                language=language,
                context=full_context,
                conversation=batches,
            )
            datasets.append(dataset)

        logger.info(
            f"Generated {len(datasets)} dataset(s) with " f"{sum(len(d.conversation) for d in datasets)} total batches"
        )

        return datasets


def _build_agentic_metadata(chunk: Chunk, generated_query: GeneratedQuery) -> dict[str, Any]:
    """Build agentic metadata for a query batch.

    Args:
        chunk: The source chunk
        generated_query: The generated query

    Returns:
        dict: Agentic metadata
    """
    metadata: dict[str, Any] = {"chunk_id": chunk.chunk_id}
    if generated_query.difficulty:
        metadata["difficulty"] = generated_query.difficulty
    if generated_query.query_type:
        metadata["query_type"] = generated_query.query_type
    return metadata


def _build_conversation_metadata(chunk: Chunk, turn: ConversationTurn) -> dict[str, Any]:
    """Build agentic metadata for a conversation turn batch.

    Args:
        chunk: The source chunk
        turn: The conversation turn

    Returns:
        dict: Agentic metadata including conversation tracking
    """
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "turn_number": turn.turn_number,
        "conversation_mode": True,
    }
    if turn.turn_number > 1:
        metadata["builds_on"] = f"{chunk.chunk_id}_t{turn.turn_number - 1}"
    if turn.difficulty:
        metadata["difficulty"] = turn.difficulty
    if turn.query_type:
        metadata["query_type"] = turn.query_type
    if turn.expected_context:
        metadata["expected_context"] = turn.expected_context
    return metadata
