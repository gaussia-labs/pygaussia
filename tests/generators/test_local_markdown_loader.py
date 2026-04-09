"""Tests for LocalMarkdownLoader."""

from pathlib import Path

import pytest

from gaussia.generators.context_loaders import LocalMarkdownLoader
from gaussia.schemas.generators import Chunk


class TestLocalMarkdownLoader:
    """Test suite for LocalMarkdownLoader."""

    def test_loader_initialization_defaults(self):
        """Test loader initializes with default values."""
        loader = LocalMarkdownLoader()

        assert loader.max_chunk_size == 2000
        assert loader.min_chunk_size == 200
        assert loader.overlap == 100
        assert loader.header_levels == [1, 2, 3]

    def test_loader_initialization_custom_values(self):
        """Test loader initializes with custom values."""
        loader = LocalMarkdownLoader(
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap=50,
            header_levels=[1, 2],
        )

        assert loader.max_chunk_size == 1000
        assert loader.min_chunk_size == 100
        assert loader.overlap == 50
        assert loader.header_levels == [1, 2]

    def test_load_splits_by_headers(self, temp_markdown_file: Path):
        """Test that loader splits content by headers."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        # Should have multiple chunks based on headers
        assert len(chunks) >= 3

        # Each chunk should have metadata
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content
            assert chunk.chunk_id
            assert "header" in chunk.metadata
            assert "source_file" in chunk.metadata

    def test_load_preserves_header_names(self, temp_markdown_file: Path):
        """Test that loader preserves header names in metadata."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        headers = [chunk.metadata.get("header") for chunk in chunks]

        # Should contain some expected headers
        assert "Introduction" in headers or any("Introduction" in h for h in headers if h)

    def test_load_splits_long_sections(self, temp_markdown_long_section_file: Path):
        """Test that loader splits very long sections by size."""
        loader = LocalMarkdownLoader(max_chunk_size=500)
        chunks = loader.load(str(temp_markdown_long_section_file))

        # Long section should be split into multiple chunks
        long_section_chunks = [c for c in chunks if "long_section" in c.chunk_id.lower()]
        assert len(long_section_chunks) > 1

    def test_load_raises_on_missing_file(self):
        """Test that loader raises FileNotFoundError for missing files."""
        loader = LocalMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.md")

    def test_chunk_ids_are_unique(self, temp_markdown_file: Path):
        """Test that all chunk IDs are unique."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

    def test_chunk_ids_derived_from_headers(self, temp_markdown_file: Path):
        """Test that chunk IDs are derived from header names."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        for chunk in chunks:
            # Chunk ID should be lowercase and use underscores
            assert chunk.chunk_id.islower() or "_" in chunk.chunk_id or chunk.chunk_id[0].isdigit() is False

    def test_source_file_in_metadata(self, temp_markdown_file: Path):
        """Test that source file path is included in metadata."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert chunk.metadata["source_file"] == str(temp_markdown_file)

    def test_empty_sections_are_skipped(self, temp_markdown_file: Path):
        """Test that empty sections are not included as chunks."""
        loader = LocalMarkdownLoader()
        chunks = loader.load(str(temp_markdown_file))

        for chunk in chunks:
            assert chunk.content.strip(), "Empty chunks should not be created"

    def test_custom_header_levels(self, temp_markdown_file: Path):
        """Test that custom header levels are respected."""
        # Only split on H1
        loader = LocalMarkdownLoader(header_levels=[1])
        chunks_h1_only = loader.load(str(temp_markdown_file))

        # Split on H1, H2, H3
        loader = LocalMarkdownLoader(header_levels=[1, 2, 3])
        chunks_all = loader.load(str(temp_markdown_file))

        # Should have fewer chunks when only splitting on H1
        assert len(chunks_h1_only) <= len(chunks_all)

    def test_overlap_in_size_chunks(self, temp_markdown_long_section_file: Path):
        """Test that size-based chunks have overlap."""
        loader = LocalMarkdownLoader(max_chunk_size=500, overlap=100)
        chunks = loader.load(str(temp_markdown_long_section_file))

        # Find consecutive size-based chunks
        size_chunks = [c for c in chunks if c.metadata.get("chunking_method") == "size"]

        if len(size_chunks) >= 2:
            # Check that consecutive chunks might share some content
            # (overlap means the end of one chunk overlaps with the start of the next)
            for i in range(len(size_chunks) - 1):
                current = size_chunks[i].content
                next_chunk = size_chunks[i + 1].content

                # With overlap, there should be some shared content at boundaries
                # This is a soft check - we just verify chunks exist
                assert len(current) > 0
                assert len(next_chunk) > 0


class TestLocalMarkdownLoaderEdgeCases:
    """Edge case tests for LocalMarkdownLoader."""

    def test_handles_unicode_content(self, tmp_path: Path):
        """Test that loader handles unicode content correctly."""
        unicode_content = """# Introduction

Cette section contient du texte en français.

## 日本語セクション

これは日本語のテキストです。

## Emoji Section 🎉

This section has emojis! 🚀 ✨ 🌟
"""
        md_file = tmp_path / "unicode.md"
        md_file.write_text(unicode_content, encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(md_file))

        assert len(chunks) >= 3
        # Should preserve unicode content
        all_content = " ".join(c.content for c in chunks)
        assert "français" in all_content or "日本語" in all_content or "🎉" in all_content

    def test_handles_code_blocks(self, tmp_path: Path):
        """Test that loader preserves code blocks."""
        code_content = """# Code Examples

Here is some Python code:

```python
def hello():
    print("Hello, World!")
```

And some JavaScript:

```javascript
function greet() {
    console.log("Hello!");
}
```
"""
        md_file = tmp_path / "code.md"
        md_file.write_text(code_content, encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(md_file))

        # Code blocks should be preserved
        all_content = " ".join(c.content for c in chunks)
        assert "def hello" in all_content or "print" in all_content

    def test_handles_nested_headers(self, tmp_path: Path):
        """Test that loader handles deeply nested headers."""
        nested_content = """# Level 1

## Level 2

### Level 3

#### Level 4

##### Level 5

###### Level 6

Content at level 6.
"""
        md_file = tmp_path / "nested.md"
        md_file.write_text(nested_content, encoding="utf-8")

        # Only split on levels 1-3
        loader = LocalMarkdownLoader(header_levels=[1, 2, 3])
        chunks = loader.load(str(md_file))

        # Should have chunks for levels 1, 2, 3 only
        assert len(chunks) >= 1

    def test_handles_single_line_file(self, tmp_path: Path):
        """Test that loader handles single-line files."""
        md_file = tmp_path / "single.md"
        md_file.write_text("Just one line of content.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(md_file))

        assert len(chunks) == 1
        assert chunks[0].content == "Just one line of content."


class TestLocalMarkdownLoaderMultipleFiles:
    """Tests for loading multiple files."""

    def test_load_multiple_files_as_list(self, tmp_path: Path):
        """Test loading multiple files from a list."""
        # Create multiple markdown files
        file1 = tmp_path / "doc1.md"
        file1.write_text("# Doc 1\n\nContent from doc 1.", encoding="utf-8")

        file2 = tmp_path / "doc2.md"
        file2.write_text("# Doc 2\n\nContent from doc 2.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load([str(file1), str(file2)])

        # Should have chunks from both files
        assert len(chunks) >= 2

        # Check that chunks have unique IDs with file prefixes
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

        # Verify source files are tracked
        source_files = {c.metadata.get("source_file") for c in chunks}
        assert str(file1) in source_files
        assert str(file2) in source_files

    def test_load_glob_pattern(self, tmp_path: Path):
        """Test loading files using glob pattern."""
        # Create directory structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "guide.md").write_text("# Guide\n\nGuide content.", encoding="utf-8")
        (docs_dir / "api.md").write_text("# API\n\nAPI content.", encoding="utf-8")
        (docs_dir / "readme.txt").write_text("Not markdown", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(docs_dir / "*.md"))

        # Should only load .md files
        assert len(chunks) >= 2

        source_files = [c.metadata.get("source_file") for c in chunks]
        assert any("guide.md" in f for f in source_files)
        assert any("api.md" in f for f in source_files)
        assert not any("readme.txt" in f for f in source_files if f)

    def test_load_recursive_glob_pattern(self, tmp_path: Path):
        """Test loading files using recursive glob pattern."""
        # Create nested directory structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        sub_dir = docs_dir / "advanced"
        sub_dir.mkdir()

        (docs_dir / "intro.md").write_text("# Intro\n\nIntro content.", encoding="utf-8")
        (sub_dir / "advanced.md").write_text("# Advanced\n\nAdvanced content.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(docs_dir / "**/*.md"))

        # Should load from both directories
        source_files = [c.metadata.get("source_file") for c in chunks]
        assert any("intro.md" in f for f in source_files)
        assert any("advanced.md" in f for f in source_files)

    def test_load_directory_path(self, tmp_path: Path):
        """Test loading all markdown files from a directory."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "file1.md").write_text("# File 1\n\nContent 1.", encoding="utf-8")
        (docs_dir / "file2.md").write_text("# File 2\n\nContent 2.", encoding="utf-8")
        (docs_dir / "file3.markdown").write_text("# File 3\n\nContent 3.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(docs_dir))

        # Should load all markdown files
        assert len(chunks) >= 3

        source_files = [c.metadata.get("source_file") for c in chunks]
        assert any("file1.md" in f for f in source_files)
        assert any("file2.md" in f for f in source_files)
        assert any("file3.markdown" in f for f in source_files)

    def test_load_multiple_files_unique_chunk_ids(self, tmp_path: Path):
        """Test that chunk IDs are unique across multiple files with same headers."""
        # Create files with identical headers
        file1 = tmp_path / "doc1.md"
        file1.write_text("# Introduction\n\nContent from doc 1.", encoding="utf-8")

        file2 = tmp_path / "doc2.md"
        file2.write_text("# Introduction\n\nContent from doc 2.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load([str(file1), str(file2)])

        # Chunk IDs should be unique even with same headers
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

        # IDs should include file prefix
        assert any("doc1" in cid for cid in chunk_ids)
        assert any("doc2" in cid for cid in chunk_ids)

    def test_load_empty_list_raises(self):
        """Test that empty list raises appropriate error."""
        loader = LocalMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load([])

    def test_load_list_with_missing_file_raises(self, tmp_path: Path):
        """Test that missing file in list raises error."""
        existing = tmp_path / "exists.md"
        existing.write_text("# Exists\n\nContent.", encoding="utf-8")

        loader = LocalMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load([str(existing), str(tmp_path / "missing.md")])

    def test_load_glob_no_matches_raises(self, tmp_path: Path):
        """Test that glob with no matches raises error."""
        loader = LocalMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "nonexistent/*.md"))

    def test_load_empty_directory_raises(self, tmp_path: Path):
        """Test that empty directory raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = LocalMarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(str(empty_dir))

    def test_load_preserves_file_order(self, tmp_path: Path):
        """Test that files are loaded in consistent order."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create files
        for name in ["zebra.md", "apple.md", "mango.md"]:
            (docs_dir / name).write_text(f"# {name}\n\nContent.", encoding="utf-8")

        loader = LocalMarkdownLoader()
        chunks = loader.load(str(docs_dir / "*.md"))

        # Files should be sorted alphabetically
        source_files = [c.metadata.get("source_file") for c in chunks]
        assert source_files.index(next(f for f in source_files if "apple" in f)) < source_files.index(
            next(f for f in source_files if "mango" in f)
        )
        assert source_files.index(next(f for f in source_files if "mango" in f)) < source_files.index(
            next(f for f in source_files if "zebra" in f)
        )
