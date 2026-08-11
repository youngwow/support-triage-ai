"""
Markdown chunking.

Two things downstream depends on and neither is obvious from the signature:
headings stay *inside* the chunk text (the embedding must see them), and ids are
handed out contiguously from ``start_id`` so a whole corpus can be numbered by
chaining calls — which is exactly what ``FaissKnowledgeBase.load`` does.
"""

from pathlib import Path

import pytest

from src.config import BASE_DIR
from src.rag.chunker import DEFAULT_CHUNK_SIZE, split_markdown


KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
CORPUS = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))

DOCUMENT = """\
# Управление платежами

## Смена банковской карты
Перейдите в «Настройки профиля» и добавьте новую карту.

## Двойные списания
Такие тикеты получают приоритет High.
"""


# --- headings ------------------------------------------------------------


def test_the_heading_stays_inside_the_chunk_text() -> None:
    chunks = split_markdown(DOCUMENT, source="payment_methods.md")

    assert "## Смена банковской карты" in chunks[0].text
    assert "## Двойные списания" in chunks[1].text


def test_the_heading_is_also_recorded_as_metadata() -> None:
    chunks = split_markdown(DOCUMENT, source="payment_methods.md")

    assert [chunk.heading for chunk in chunks] == [
        "Смена банковской карты",
        "Двойные списания",
    ]


@pytest.mark.parametrize(
    ("document", "expected_heading"),
    [
        ("# Только h1\nтекст", "Только h1"),
        ("# H1\n## H2\nтекст", "H2"),
        ("# H1\n## H2\n### H3\nтекст", "H3"),
        ("текст без заголовка", ""),
    ],
    ids=["h1_only", "h2_wins_over_h1", "h3_wins_over_h2", "no_heading_at_all"],
)
def test_the_deepest_heading_present_labels_the_chunk(
    document: str, expected_heading: str
) -> None:
    chunks = split_markdown(document, source="doc.md")

    assert [chunk.heading for chunk in chunks] == [expected_heading]


# --- sizing --------------------------------------------------------------


def test_a_section_that_fits_is_left_whole() -> None:
    section = "## Раздел\n" + "а" * 100

    chunks = split_markdown(section, source="doc.md", chunk_size=len(section))

    assert len(chunks) == 1
    assert chunks[0].text == section


def test_an_oversized_section_is_split_into_several_chunks() -> None:
    section = "## Раздел\n" + "а" * 100
    chunk_size = 40

    chunks = split_markdown(
        section, source="doc.md", chunk_size=chunk_size, chunk_overlap=0
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= chunk_size for chunk in chunks)
    assert [chunk.id for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.heading for chunk in chunks} == {"Раздел"}


# --- ids -----------------------------------------------------------------


@pytest.mark.parametrize("start_id", [0, 1, 42], ids=["from_zero", "from_one", "from_42"])
def test_ids_are_contiguous_from_start_id(start_id: int) -> None:
    chunks = split_markdown(DOCUMENT, source="doc.md", start_id=start_id)

    assert [chunk.id for chunk in chunks] == list(
        range(start_id, start_id + len(chunks))
    )


def test_chaining_documents_keeps_ids_unique_across_the_corpus() -> None:
    first = split_markdown(DOCUMENT, source="a.md", start_id=0)
    second = split_markdown(DOCUMENT, source="b.md", start_id=len(first))

    all_ids = [chunk.id for chunk in first + second]
    assert all_ids == list(range(len(all_ids)))


# --- empty input ---------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    ["", "   ", "\n\n", "\n   \n\t\n"],
    ids=["empty", "spaces", "newlines", "mixed_whitespace"],
)
def test_a_document_with_no_content_produces_no_chunks(document: str) -> None:
    assert split_markdown(document, source="doc.md") == []


def test_no_chunk_is_blank_or_carries_untrimmed_whitespace() -> None:
    document = "## Раздел\n\n\n   \nТекст.\n\n\n## Пустой раздел\n\n\n"

    chunks = split_markdown(document, source="doc.md")

    assert chunks
    assert all(chunk.text == chunk.text.strip() != "" for chunk in chunks)


# --- the real corpus -----------------------------------------------------


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.name)
def test_every_chunk_of_the_real_corpus_is_usable(path: Path) -> None:
    chunks = split_markdown(path.read_text(encoding="utf-8"), source=path.name)

    assert chunks, f"{path.name} produced no chunks"
    for chunk in chunks:
        assert chunk.text.strip() == chunk.text != ""
        assert chunk.source == path.name
        assert len(chunk.text) <= DEFAULT_CHUNK_SIZE


def test_the_corpus_is_present() -> None:
    # Guards the parametrisation above: an empty CORPUS would silently collect
    # zero tests instead of failing.
    assert {path.name for path in CORPUS} == {
        "account_security.md",
        "incident_management.md",
        "payment_methods.md",
    }
