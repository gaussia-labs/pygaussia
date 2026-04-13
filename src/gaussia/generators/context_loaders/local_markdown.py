"""Local markdown file context loader with hybrid chunking."""

import glob as glob_module
import re
from pathlib import Path

from loguru import logger

from gaussia.schemas.generators import BaseContextLoader, Chunk


class LocalMarkdownLoader(BaseContextLoader):
    """Context loader for local markdown files with hybrid chunking.

    Uses a hybrid strategy:
    1. Primary: Split by markdown headers (H1, H2, H3, etc.)
    2. Fallback: Split by character count for long sections without headers

    Supports loading from:
    - Single file path: "docs/guide.md"
    - Multiple file paths: ["docs/guide.md", "docs/api.md"]
    - Glob patterns: "docs/**/*.md"

    Args:
        max_chunk_size: Maximum characters per chunk (default: 2000)
        min_chunk_size: Minimum characters per chunk (default: 200)
        overlap: Character overlap between size-based chunks (default: 100)
        header_levels: Header levels to split on (default: [1, 2, 3])
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 200,
        overlap: int = 100,
        header_levels: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.header_levels = header_levels or [1, 2, 3]

    def _split_by_headers(self, content: str) -> list[tuple[str, str]]:
        """Split content by markdown headers.

        Args:
            content: Full markdown content

        Returns:
            list[tuple[str, str]]: List of (header, section_content) tuples
        """
        sections = []
        current_header = "Introduction"
        current_content: list[str] = []

        lines = content.split("\n")
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                level = len(header_match.group(1))
                if level in self.header_levels:
                    # Save previous section
                    if current_content:
                        sections.append((current_header, "\n".join(current_content)))
                    current_header = header_match.group(2).strip()
                    current_content = []
                else:
                    current_content.append(line)
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            sections.append((current_header, "\n".join(current_content)))

        return sections

    def _split_by_size(self, content: str, base_id: str) -> list[Chunk]:
        """Split content by character size with overlap.

        Args:
            content: Text content to split
            base_id: Base ID for chunk naming

        Returns:
            list[Chunk]: Size-based chunks
        """
        chunks = []
        start = 0
        part = 1

        while start < len(content):
            end = start + self.max_chunk_size

            # Try to break at a sentence or paragraph boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + self.min_chunk_size:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sentence_break = content.rfind(". ", start, end)
                    if sentence_break > start + self.min_chunk_size:
                        end = sentence_break + 2

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        chunk_id=f"{base_id}_part{part}",
                        metadata={"chunking_method": "size"},
                    )
                )
                part += 1

            start = end - self.overlap if end < len(content) else end

        return chunks

    def _load_single_file(self, file_path: Path) -> list[Chunk]:
        """Load and chunk a single markdown file.

        Args:
            file_path: Path object to the markdown file

        Returns:
            list[Chunk]: Chunked content from this file
        """
        if file_path.suffix.lower() not in [".md", ".markdown"]:
            logger.warning(f"File {file_path} may not be markdown format")

        logger.info(f"Loading markdown file: {file_path}")
        content = file_path.read_text(encoding="utf-8")

        # First, try header-based splitting
        sections = self._split_by_headers(content)

        chunks = []
        for i, (header, section_content) in enumerate(sections):
            section_content = section_content.strip()
            if not section_content:
                continue

            # Create base ID from header (include file stem for uniqueness)
            base_id = re.sub(r"[^a-zA-Z0-9]+", "_", header.lower()).strip("_")
            if not base_id:
                base_id = f"section_{i + 1}"

            # Prefix with file stem to ensure unique chunk IDs across files
            file_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", file_path.stem.lower()).strip("_")
            full_chunk_id = f"{file_prefix}_{base_id}"

            # If section is too long, apply size-based splitting
            if len(section_content) > self.max_chunk_size:
                logger.debug(
                    f"Section '{header}' ({len(section_content)} chars) " f"exceeds max size, splitting further"
                )
                sub_chunks = self._split_by_size(section_content, full_chunk_id)
                for sub_chunk in sub_chunks:
                    if sub_chunk.metadata is not None:
                        sub_chunk.metadata["header"] = header
                        sub_chunk.metadata["source_file"] = str(file_path)
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    Chunk(
                        content=section_content,
                        chunk_id=full_chunk_id,
                        metadata={
                            "header": header,
                            "chunking_method": "header",
                            "source_file": str(file_path),
                        },
                    )
                )

        # If no header-based chunks were created, fall back to pure size-based
        if not chunks:
            logger.info("No header-based chunks created, using size-based chunking")
            file_prefix = re.sub(r"[^a-zA-Z0-9]+", "_", file_path.stem.lower()).strip("_")
            chunks = self._split_by_size(content, file_prefix)
            for chunk in chunks:
                if chunk.metadata is not None:
                    chunk.metadata["source_file"] = str(file_path)

        return chunks

    def _resolve_paths(self, source: str | list[str]) -> list[Path]:
        """Resolve source to a list of file paths.

        Args:
            source: Single path, list of paths, or glob pattern

        Returns:
            list[Path]: Resolved file paths

        Raises:
            FileNotFoundError: If no files are found
        """
        paths: list[Path] = []

        if isinstance(source, list):
            # Multiple explicit paths
            if not source:
                raise FileNotFoundError("No files provided (empty list)")
            for s in source:
                p = Path(s)
                if not p.exists():
                    raise FileNotFoundError(f"Markdown file not found: {s}")
                paths.append(p)
        else:
            # Single path or glob pattern
            path = Path(source)
            if path.exists() and path.is_file():
                # Single existing file
                paths.append(path)
            elif "*" in source or "?" in source:
                # Glob pattern
                matched = glob_module.glob(source, recursive=True)
                if not matched:
                    raise FileNotFoundError(f"No files found matching pattern: {source}")
                paths.extend(Path(m) for m in sorted(matched))
            elif path.exists() and path.is_dir():
                # Directory - load all markdown files
                md_files = list(path.glob("**/*.md")) + list(path.glob("**/*.markdown"))
                if not md_files:
                    raise FileNotFoundError(f"No markdown files found in directory: {source}")
                paths.extend(sorted(md_files))
            else:
                raise FileNotFoundError(f"Markdown file not found: {source}")

        return paths

    def load(self, source: str | list[str]) -> list[Chunk]:
        """Load and chunk markdown files.

        Supports multiple input formats:
        - Single file path: "docs/guide.md"
        - Multiple file paths: ["docs/guide.md", "docs/api.md"]
        - Glob patterns: "docs/**/*.md" or "docs/*.md"
        - Directory path: "docs/" (loads all .md files recursively)

        Args:
            source: Path(s) to markdown file(s), glob pattern, or directory

        Returns:
            list[Chunk]: Chunked content from all files

        Raises:
            FileNotFoundError: If no markdown files are found

        Examples:
            >>> loader = LocalMarkdownLoader()
            >>> # Single file
            >>> chunks = loader.load("docs/guide.md")
            >>> # Multiple files
            >>> chunks = loader.load(["docs/guide.md", "docs/api.md"])
            >>> # Glob pattern
            >>> chunks = loader.load("docs/**/*.md")
            >>> # Directory
            >>> chunks = loader.load("docs/")
        """
        paths = self._resolve_paths(source)

        logger.info(f"Loading {len(paths)} markdown file(s)")

        all_chunks: list[Chunk] = []
        for file_path in paths:
            file_chunks = self._load_single_file(file_path)
            all_chunks.extend(file_chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(paths)} file(s)")
        return all_chunks


__all__ = ["LocalMarkdownLoader"]
