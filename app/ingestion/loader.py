from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class DocumentPage:
    page_number: int
    text: str
    source: str


def load_pdf(pdf_path: str | Path) -> list[DocumentPage]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)

    pages: list[DocumentPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=text.strip(),
                source=path.name,
            )
        )

    return pages
