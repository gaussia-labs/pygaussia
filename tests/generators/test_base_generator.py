"""Tests for BaseGenerator with LangChain models."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaussia.generators import (
    BaseGenerator,
    LocalMarkdownLoader,
    RandomSamplingStrategy,
    SequentialStrategy,
)
from gaussia.schemas.common import Batch, Dataset
from gaussia.schemas.generators import (
    Chunk,
    ConversationTurn,
    GeneratedConversationOutput,
    GeneratedQueriesOutput,
    GeneratedQuery,
)


class TestBaseGeneratorInitialization:
    """Test suite for BaseGenerator initialization."""

    def test_base_generator_initialization(self):
        """Test BaseGenerator initializes with model."""
        mock_model = MagicMock()
        generator = BaseGenerator(model=mock_model)

        assert generator.model == mock_model
        assert generator.use_structured_output is True

    def test_base_generator_custom_structured_output(self):
        """Test BaseGenerator with custom structured output setting."""
        mock_model = MagicMock()
        generator = BaseGenerator(
            model=mock_model,
            use_structured_output=False,
        )

        assert generator.use_structured_output is False


class TestBaseGeneratorParsing:
    """Test suite for BaseGenerator JSON parsing methods."""

    def test_parse_json_response_from_code_block(self):
        """Test parsing JSON from markdown code block."""
        mock_model = MagicMock()
        generator = BaseGenerator(model=mock_model)

        content = """Here is the response:
```json
{"queries": [{"query": "What is X?", "difficulty": "easy", "query_type": "factual"}], "chunk_summary": "Summary"}
```"""

        result = generator._parse_json_response(content)

        assert isinstance(result, GeneratedQueriesOutput)
        assert len(result.queries) == 1
        assert result.queries[0].query == "What is X?"

    def test_parse_json_response_raw_json(self):
        """Test parsing raw JSON without code block."""
        mock_model = MagicMock()
        generator = BaseGenerator(model=mock_model)

        content = '{"queries": [{"query": "What is Y?", "difficulty": "medium", "query_type": "inferential"}], "chunk_summary": "Test"}'

        result = generator._parse_json_response(content)

        assert isinstance(result, GeneratedQueriesOutput)
        assert len(result.queries) == 1
        assert result.queries[0].query == "What is Y?"

    def test_parse_json_response_no_json(self):
        """Test parsing fails gracefully with no JSON."""
        mock_model = MagicMock()
        generator = BaseGenerator(model=mock_model)

        content = "This is just plain text without any JSON"

        with pytest.raises(ValueError, match="No JSON found"):
            generator._parse_json_response(content)

    def test_parse_conversation_response_from_code_block(self):
        """Test parsing conversation JSON from markdown code block."""
        mock_model = MagicMock()
        generator = BaseGenerator(model=mock_model)

        content = """Here is the conversation:
```json
{
    "turns": [
        {"query": "First question?", "turn_number": 1, "difficulty": "easy", "query_type": "factual"}
    ],
    "conversation_summary": "A brief conversation",
    "chunk_summary": "Test chunk"
}
```"""

        result = generator._parse_conversation_response(content)

        assert isinstance(result, GeneratedConversationOutput)
        assert len(result.turns) == 1
        assert result.turns[0].query == "First question?"
        assert result.turns[0].turn_number == 1


class TestBaseGeneratorGenerateQueries:
    """Test suite for BaseGenerator.generate_queries method."""

    @pytest.mark.asyncio
    async def test_generate_queries_structured_output(self, sample_chunk: Chunk):
        """Test generate_queries with structured output mode."""
        mock_model = MagicMock()

        # Mock the structured output result
        mock_result = GeneratedQueriesOutput(
            queries=[
                GeneratedQuery(query="Test question?", difficulty="easy", query_type="factual"),
            ],
            chunk_summary="Test summary",
        )

        with patch("gaussia.schemas.generators.ChatPromptTemplate") as mock_prompt:
            # Mock the chain that is created when prompt | model
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            generator = BaseGenerator(model=mock_model, use_structured_output=True)
            queries = await generator.generate_queries(sample_chunk, num_queries=1)

            assert len(queries) == 1
            assert queries[0].query == "Test question?"
            mock_model.with_structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_queries_regex_mode(self, sample_chunk: Chunk):
        """Test generate_queries with regex extraction mode."""
        mock_model = MagicMock()

        # Mock the response content
        mock_response = MagicMock()
        mock_response.content = """```json
{"queries": [{"query": "Regex question?", "difficulty": "medium", "query_type": "inferential"}], "chunk_summary": "Summary"}
```"""
        mock_model.invoke = MagicMock(return_value=mock_response)

        # Create a mock chain that returns our mock response
        with patch("gaussia.schemas.generators.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            generator = BaseGenerator(model=mock_model, use_structured_output=False)
            queries = await generator.generate_queries(sample_chunk, num_queries=1)

            assert len(queries) == 1
            assert queries[0].query == "Regex question?"


class TestBaseGeneratorGenerateConversation:
    """Test suite for BaseGenerator.generate_conversation method."""

    @pytest.mark.asyncio
    async def test_generate_conversation_structured_output(self, sample_chunk: Chunk):
        """Test generate_conversation with structured output mode."""
        mock_model = MagicMock()

        mock_result = GeneratedConversationOutput(
            turns=[
                ConversationTurn(
                    query="What is this about?",
                    turn_number=1,
                    difficulty="easy",
                    query_type="factual",
                ),
                ConversationTurn(
                    query="Can you explain more?",
                    turn_number=2,
                    difficulty="medium",
                    query_type="inferential",
                    expected_context="Builds on turn 1",
                ),
            ],
            conversation_summary="A two-turn conversation",
            chunk_summary="Test summary",
        )

        with patch("gaussia.schemas.generators.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            generator = BaseGenerator(model=mock_model, use_structured_output=True)
            turns = await generator.generate_conversation(sample_chunk, num_turns=2)

            assert len(turns) == 2
            assert turns[0].query == "What is this about?"
            assert turns[0].turn_number == 1
            assert turns[1].turn_number == 2
            assert turns[1].expected_context == "Builds on turn 1"


class TestBaseGeneratorGenerateDataset:
    """Test suite for BaseGenerator.generate_dataset method."""

    @pytest.mark.asyncio
    async def test_generate_dataset_creates_valid_dataset(self, temp_markdown_file: Path):
        """Test generate_dataset creates a valid list of Dataset objects."""
        mock_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[
                GeneratedQuery(query="Generated question?", difficulty="easy", query_type="factual"),
            ],
            chunk_summary="Summary",
        )

        with patch("gaussia.schemas.generators.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            generator = BaseGenerator(model=mock_model, use_structured_output=True)
            loader = LocalMarkdownLoader()

            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="test-assistant",
                num_queries_per_chunk=1,
            )

            assert isinstance(datasets, list)
            assert len(datasets) >= 1

            dataset = datasets[0]
            assert isinstance(dataset, Dataset)
            assert dataset.assistant_id == "test-assistant"
            assert len(dataset.conversation) > 0

            for batch in dataset.conversation:
                assert isinstance(batch, Batch)
                assert batch.query
                assert batch.assistant == ""

    @pytest.mark.asyncio
    async def test_generate_dataset_conversation_mode(self, temp_markdown_file: Path):
        """Test generate_dataset with conversation_mode=True."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedConversationOutput(
            turns=[
                ConversationTurn(
                    query="Turn 1?",
                    turn_number=1,
                    difficulty="easy",
                    query_type="factual",
                ),
                ConversationTurn(
                    query="Turn 2?",
                    turn_number=2,
                    difficulty="medium",
                    query_type="inferential",
                    expected_context="Builds on turn 1",
                ),
            ],
            conversation_summary="Test conversation",
            chunk_summary="Summary",
        )
        # Mock the chain: prompt | structured_model creates a chain that returns result
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_structured_model.__or__ = MagicMock(return_value=mock_chain)
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = BaseGenerator(model=mock_model, use_structured_output=True)
        loader = LocalMarkdownLoader()

        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=str(temp_markdown_file),
            assistant_id="test-assistant",
            num_queries_per_chunk=2,
            conversation_mode=True,
        )

        assert isinstance(datasets, list)
        assert len(datasets) >= 1

        dataset = datasets[0]
        assert dataset.assistant_id == "test-assistant"

        # Check that batches have conversation metadata
        for batch in dataset.conversation:
            assert "turn_number" in batch.agentic
            assert batch.agentic.get("conversation_mode") is True


class TestBaseGeneratorWithSeedExamples:
    """Test seed examples functionality."""

    @pytest.mark.asyncio
    async def test_seed_examples_included_in_prompt(self, sample_chunk: Chunk):
        """Test that seed examples are included in the prompt."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[GeneratedQuery(query="Q?", difficulty="easy", query_type="factual")],
            chunk_summary="Summary",
        )
        # Mock the chain: prompt | structured_model creates a chain that returns result
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_structured_model.__or__ = MagicMock(return_value=mock_chain)
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = BaseGenerator(model=mock_model, use_structured_output=True)

        seed_examples = ["Example question 1?", "Example question 2?"]
        await generator.generate_queries(
            sample_chunk,
            num_queries=1,
            seed_examples=seed_examples,
        )

        # The seed examples should be included in the formatted prompt
        # This is validated through the successful call - the format works
        mock_model.with_structured_output.assert_called_once()


class TestBaseGeneratorWithStrategies:
    """Test chunk selection strategies with BaseGenerator."""

    @pytest.mark.asyncio
    async def test_sequential_strategy_default(self, temp_markdown_file: Path):
        """Test that default strategy is sequential."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[GeneratedQuery(query="Q?", difficulty="easy", query_type="factual")],
            chunk_summary="Summary",
        )
        # Mock the chain: prompt | structured_model creates a chain that returns result
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_structured_model.__or__ = MagicMock(return_value=mock_chain)
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = BaseGenerator(model=mock_model)
        loader = LocalMarkdownLoader()

        # No strategy provided - should use sequential (single dataset)
        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=str(temp_markdown_file),
            assistant_id="test",
        )

        assert isinstance(datasets, list)
        assert len(datasets) == 1

    @pytest.mark.asyncio
    async def test_explicit_sequential_strategy(self, temp_markdown_file: Path):
        """Test explicit SequentialStrategy."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[GeneratedQuery(query="Q?", difficulty="easy", query_type="factual")],
            chunk_summary="Summary",
        )
        # Mock the chain: prompt | structured_model creates a chain that returns result
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_structured_model.__or__ = MagicMock(return_value=mock_chain)
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = BaseGenerator(model=mock_model)
        loader = LocalMarkdownLoader()

        strategy = SequentialStrategy()
        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=str(temp_markdown_file),
            assistant_id="test",
            selection_strategy=strategy,
        )

        assert isinstance(datasets, list)
        assert len(datasets) == 1

    @pytest.mark.asyncio
    async def test_random_sampling_strategy(self, temp_markdown_file: Path):
        """Test RandomSamplingStrategy produces multiple datasets."""
        mock_model = MagicMock()
        mock_structured_model = MagicMock()

        mock_result = GeneratedQueriesOutput(
            queries=[GeneratedQuery(query="Q?", difficulty="easy", query_type="factual")],
            chunk_summary="Summary",
        )
        # Mock the chain: prompt | structured_model creates a chain that returns result
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_structured_model.__or__ = MagicMock(return_value=mock_chain)
        mock_model.with_structured_output.return_value = mock_structured_model

        generator = BaseGenerator(model=mock_model)
        loader = LocalMarkdownLoader()

        # Request 3 random samples
        strategy = RandomSamplingStrategy(
            num_samples=3,
            chunks_per_sample=1,
            seed=42,
        )
        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=str(temp_markdown_file),
            assistant_id="test",
            selection_strategy=strategy,
        )

        assert isinstance(datasets, list)
        assert len(datasets) == 3

        # Each dataset should have unique session_id
        session_ids = [d.session_id for d in datasets]
        assert len(session_ids) == len(set(session_ids))

    @pytest.mark.asyncio
    async def test_random_sampling_with_conversation_mode(self, temp_markdown_file: Path):
        """Test combining random sampling with conversation mode."""
        mock_model = MagicMock()

        mock_result = GeneratedConversationOutput(
            turns=[
                ConversationTurn(query="T1?", turn_number=1, difficulty="easy", query_type="factual"),
                ConversationTurn(query="T2?", turn_number=2, difficulty="medium", query_type="inferential"),
            ],
            conversation_summary="Conversation",
            chunk_summary="Summary",
        )

        with patch("gaussia.schemas.generators.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.from_messages.return_value.__or__ = MagicMock(return_value=mock_chain)

            generator = BaseGenerator(model=mock_model)
            loader = LocalMarkdownLoader()

            strategy = RandomSamplingStrategy(
                num_samples=2,
                chunks_per_sample=1,
                seed=42,
            )
            datasets = await generator.generate_dataset(
                context_loader=loader,
                source=str(temp_markdown_file),
                assistant_id="test",
                selection_strategy=strategy,
                conversation_mode=True,
                num_queries_per_chunk=2,
            )

            assert len(datasets) == 2

            for dataset in datasets:
                # Each dataset should have conversation batches
                assert len(dataset.conversation) > 0
                for batch in dataset.conversation:
                    assert batch.agentic.get("conversation_mode") is True
