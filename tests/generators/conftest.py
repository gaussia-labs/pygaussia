"""Fixtures for generator tests."""

import tempfile
from pathlib import Path

import pytest

from gaussia.schemas.generators import Chunk, GeneratedQuery


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Fixture providing sample chunks."""
    return [
        Chunk(
            content="Gaussia is a community-driven AI evaluation framework. It provides tools for measuring fairness, quality, and safety of AI assistant responses.",
            chunk_id="about_gaussia",
            metadata={"header": "About Gaussia", "chunking_method": "header"},
        ),
        Chunk(
            content="The seven principles guide all decisions in the platform: transparency, fairness, accountability, privacy, security, reliability, and interpretability.",
            chunk_id="principles",
            metadata={"header": "Principles", "chunking_method": "header"},
        ),
    ]


@pytest.fixture
def sample_chunk() -> Chunk:
    """Fixture providing a single sample chunk."""
    return Chunk(
        content="Gaussia is a performance-measurement library for evaluating AI models. It provides metrics for fairness, toxicity, bias, and conversational quality.",
        chunk_id="gaussia_intro",
        metadata={"header": "Introduction", "chunking_method": "header"},
    )


@pytest.fixture
def sample_generated_queries() -> list[GeneratedQuery]:
    """Fixture providing sample generated queries."""
    return [
        GeneratedQuery(
            query="What is the main purpose of Gaussia?",
            difficulty="easy",
            query_type="factual",
        ),
        GeneratedQuery(
            query="How does Gaussia evaluate AI models for bias?",
            difficulty="medium",
            query_type="inferential",
        ),
        GeneratedQuery(
            query="Compare the toxicity and fairness metrics in Gaussia.",
            difficulty="hard",
            query_type="comparative",
        ),
    ]


@pytest.fixture
def sample_markdown_content() -> str:
    """Fixture providing sample markdown content."""
    return """# Introduction

This is an introduction to our platform. It provides powerful tools for AI evaluation.

## Features

Our platform has many features including:
- Feature A: Automated testing
- Feature B: Metric collection
- Feature C: Report generation

## Getting Started

Follow these steps to begin using the platform.

### Prerequisites

You need Python 3.11+ installed on your system.

### Installation

Run the following command to install:
```
pip install gaussia
```

## Advanced Usage

For advanced users, there are additional configuration options available.
"""


@pytest.fixture
def sample_markdown_no_headers() -> str:
    """Fixture providing markdown content without headers."""
    return """This is a plain text document without any markdown headers.

It contains multiple paragraphs of content that should be chunked by size.

The content discusses various topics but doesn't use any heading structure.

Each paragraph provides different information about the subject matter.
"""


@pytest.fixture
def sample_markdown_long_section() -> str:
    """Fixture providing markdown with a very long section."""
    long_content = "This is a very long paragraph. " * 200
    return f"""# Short Section

This is a short introduction.

## Long Section

{long_content}

## Another Short Section

This is the conclusion.
"""


@pytest.fixture
def temp_markdown_file(sample_markdown_content: str):
    """Fixture providing a temporary markdown file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(sample_markdown_content)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_markdown_no_headers_file(sample_markdown_no_headers: str):
    """Fixture providing a temporary markdown file without headers."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(sample_markdown_no_headers)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_markdown_long_section_file(sample_markdown_long_section: str):
    """Fixture providing a temporary markdown file with a long section."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(sample_markdown_long_section)
        f.flush()
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mock_llm_response() -> dict:
    """Fixture providing mock LLM response in JSON format."""
    return {
        "queries": [
            {"query": "What is the main purpose?", "difficulty": "easy", "query_type": "factual"},
            {"query": "How does it work?", "difficulty": "medium", "query_type": "inferential"},
        ],
        "chunk_summary": "Introduction to the platform.",
    }


@pytest.fixture
def sample_conversation_turns():
    """Fixture providing sample conversation turns."""
    from gaussia.schemas.generators import ConversationTurn

    return [
        ConversationTurn(
            query="What is machine learning?",
            turn_number=1,
            difficulty="easy",
            query_type="factual",
            expected_context=None,
        ),
        ConversationTurn(
            query="How does supervised learning differ from unsupervised?",
            turn_number=2,
            difficulty="medium",
            query_type="comparative",
            expected_context="Builds on the definition of ML from turn 1",
        ),
        ConversationTurn(
            query="When would you choose unsupervised over supervised learning?",
            turn_number=3,
            difficulty="hard",
            query_type="analytical",
            expected_context="References both supervised and unsupervised from turn 2",
        ),
    ]


@pytest.fixture
def mock_conversation_response() -> str:
    """Fixture providing mock conversation generation response string."""
    return """```json
{
    "turns": [
        {"query": "What is the main topic?", "turn_number": 1, "difficulty": "easy", "query_type": "factual", "expected_context": null},
        {"query": "Can you explain more details?", "turn_number": 2, "difficulty": "medium", "query_type": "inferential", "expected_context": "Builds on turn 1"},
        {"query": "How does this compare to alternatives?", "turn_number": 3, "difficulty": "hard", "query_type": "comparative", "expected_context": "References turns 1 and 2"}
    ],
    "conversation_summary": "A conversation exploring the main topic in depth",
    "chunk_summary": "Introduction to the platform"
}
```"""
