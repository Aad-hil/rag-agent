import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving document structure.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix only explicit hyphenated line breaks.
    # Example:
    # generative-
    # ai
    # -> generative-ai
    text = re.sub(r"(?<=\w)-\n(?=\w)", "-", text)

    # Remove trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
