"""Resume / checkpoint manager — tracks completed chapters and chunks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ChunkStatus:
    chunk_index: int
    sentence_range: tuple[int, int]  # (start, end) indices
    audio_file: str = ""
    completed: bool = False
    timestamp: float = 0.0


@dataclass
class ChapterStatus:
    chapter_index: int
    chapter_title: str = ""
    total_chunks: int = 0
    completed_chunks: int = 0
    audio_file: str = ""  # Final assembled chapter audio
    completed: bool = False
    chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CheckpointState:
    book_title: str = ""
    input_file: str = ""
    input_hash: str = ""  # SHA-256 of input file for change detection
    total_chapters: int = 0
    completed_chapters: int = 0
    m4b_assembled: bool = False
    chapters: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = 0.0
    last_updated: float = 0.0


class CheckpointManager:
    """Manages pipeline state for resumable processing."""

    def __init__(self, checkpoint_path: str | Path):
        self.path = Path(checkpoint_path)
        self.state = CheckpointState()
        self._load()

    def _load(self) -> None:
        """Load checkpoint from disk if it exists."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.state = CheckpointState(**{
                    k: v for k, v in data.items()
                    if k in CheckpointState.__dataclass_fields__
                })
            except (json.JSONDecodeError, TypeError):
                self.state = CheckpointState()

    def _save(self) -> None:
        """Persist checkpoint to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state.last_updated = time.time()
        with open(self.path, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def initialize(self, book_title: str, input_file: str, input_hash: str, chapters: list[str]) -> None:
        """Initialize checkpoint for a new book. Preserves existing progress if input unchanged."""
        if self.state.input_hash == input_hash and self.state.total_chapters == len(chapters):
            return  # Same input — resume

        self.state = CheckpointState(
            book_title=book_title,
            input_file=input_file,
            input_hash=input_hash,
            total_chapters=len(chapters),
            started_at=time.time(),
            chapters=[
                asdict(ChapterStatus(chapter_index=i, chapter_title=title))
                for i, title in enumerate(chapters)
            ],
        )
        self._save()

    def is_chapter_done(self, chapter_index: int) -> bool:
        """Check if a chapter has been fully processed."""
        if chapter_index < len(self.state.chapters):
            return self.state.chapters[chapter_index].get("completed", False)
        return False

    def is_chunk_done(self, chapter_index: int, chunk_index: int) -> bool:
        """Check if a specific chunk within a chapter is done."""
        if chapter_index < len(self.state.chapters):
            chapter = self.state.chapters[chapter_index]
            chunks = chapter.get("chunks", [])
            for chunk in chunks:
                if chunk.get("chunk_index") == chunk_index and chunk.get("completed"):
                    return True
        return False

    def mark_chunk_done(self, chapter_index: int, chunk_index: int, audio_file: str,
                        sentence_range: tuple[int, int]) -> None:
        """Mark a chunk as completed."""
        if chapter_index >= len(self.state.chapters):
            return

        chapter = self.state.chapters[chapter_index]
        chunks = chapter.get("chunks", [])

        # Update or add chunk
        found = False
        for chunk in chunks:
            if chunk.get("chunk_index") == chunk_index:
                chunk["completed"] = True
                chunk["audio_file"] = audio_file
                chunk["timestamp"] = time.time()
                found = True
                break
        if not found:
            chunks.append({
                "chunk_index": chunk_index,
                "sentence_range": list(sentence_range),
                "audio_file": audio_file,
                "completed": True,
                "timestamp": time.time(),
            })

        chapter["chunks"] = chunks
        chapter["completed_chunks"] = sum(1 for c in chunks if c.get("completed"))
        self._save()

    def mark_chapter_done(self, chapter_index: int, audio_file: str) -> None:
        """Mark a chapter as fully completed."""
        if chapter_index >= len(self.state.chapters):
            return

        chapter = self.state.chapters[chapter_index]
        chapter["completed"] = True
        chapter["audio_file"] = audio_file
        self.state.completed_chapters = sum(
            1 for ch in self.state.chapters if ch.get("completed")
        )
        self._save()

    def mark_m4b_done(self) -> None:
        """Mark the final M4B assembly as complete."""
        self.state.m4b_assembled = True
        self._save()

    def get_progress(self) -> dict[str, Any]:
        """Return a summary of current progress."""
        return {
            "book_title": self.state.book_title,
            "total_chapters": self.state.total_chapters,
            "completed_chapters": self.state.completed_chapters,
            "m4b_assembled": self.state.m4b_assembled,
            "percent": (
                round(self.state.completed_chapters / self.state.total_chapters * 100, 1)
                if self.state.total_chapters > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """Clear checkpoint and start fresh."""
        self.state = CheckpointState()
        if self.path.exists():
            self.path.unlink()
