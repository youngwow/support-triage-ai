from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.models.domain import DocumentChunk


HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def split_markdown(
    text: str,
    *,
    source: str,
    start_id: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split one markdown document into retrieval-sized chunks.

    Sections come from the header structure (headers stay in the text so the
    embedding sees them); only oversized sections fall through to the
    character splitter.
    """
    header_splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON, strip_headers=False)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[DocumentChunk] = []
    next_id = start_id
    for section in header_splitter.split_text(text):
        heading = (
            section.metadata.get("h3")
            or section.metadata.get("h2")
            or section.metadata.get("h1")
        )
        if len(section.page_content) <= chunk_size:
            pieces = [section.page_content]
        else:
            pieces = recursive_splitter.split_text(section.page_content)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                DocumentChunk(id=next_id, source=source, heading=heading, text=piece)
            )
            next_id += 1
    return chunks
