import re
from dataclasses import dataclass

from app.ingestion.cleaner import clean_text
from app.ingestion.loader import DocumentPage


@dataclass
class DocumentChunk:
    chunk_index: int
    text: str
    source: str
    page_number: int


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences while keeping the sentence-ending punctuation.
    """
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text.strip(),
    )

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_long_text(text: str, max_length: int) -> list[str]:
    """
    Split text into smaller pieces when sentence-level splitting
    still produces pieces that are too large.
    """
    words = text.split()

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        additional_length = len(word) + (1 if current else 0)

        if current and current_length + additional_length > max_length:
            pieces.append(" ".join(current))
            current = []
            current_length = 0

        current.append(word)
        current_length += additional_length

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Create semantic-ish chunks while preserving page boundaries.

    Strategy:
    - Preserve paragraphs where possible.
    - Split large paragraphs into sentences.
    - Fall back to word-based splitting for very large text.
    - Keep chunks close to the target size.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        text = clean_text(page.text)

        if not text:
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

        units: list[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                units.append(paragraph)
                continue

            sentences = _split_into_sentences(paragraph)

            if not sentences:
                units.extend(_split_long_text(paragraph, chunk_size))
                continue

            for sentence in sentences:
                if len(sentence) <= chunk_size:
                    units.append(sentence)
                else:
                    units.extend(_split_long_text(sentence, chunk_size))

        current_units: list[str] = []
        current_length = 0

        for unit in units:
            additional_length = len(unit) + (2 if current_units else 0)

            if (
                current_units
                and current_length + additional_length > chunk_size
            ):
                chunk_text = "\n\n".join(current_units)

                chunks.append(
                    DocumentChunk(
                        chunk_index=chunk_index,
                        text=chunk_text,
                        source=page.source,
                        page_number=page.page_number,
                    )
                )

                chunk_index += 1

                # Keep recent units for semantic overlap.
                overlap_units: list[str] = []
                overlap_length = 0

                for previous in reversed(current_units):
                    if overlap_length + len(previous) > overlap:
                        break

                    overlap_units.insert(0, previous)
                    overlap_length += len(previous) + 2

                current_units = overlap_units
                current_length = overlap_length

            current_units.append(unit)
            current_length += additional_length

        if current_units:
            chunks.append(
                DocumentChunk(
                    chunk_index=chunk_index,
                    text="\n\n".join(current_units),
                    source=page.source,
                    page_number=page.page_number,
                )
            )

            chunk_index += 1

    return chunks