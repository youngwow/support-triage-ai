"""Tests for src/rag/chunker.py — pure markdown → DocumentChunk splitting."""

import pytest

from src.config import BASE_DIR
from src.rag.chunker import DEFAULT_CHUNK_SIZE, split_markdown


DATA_DIR = BASE_DIR / "data"
CORPUS_PATHS = sorted(DATA_DIR.glob("*.md"))

SYNTHETIC = """# Title
intro text

## Section
section text

### Subsection
sub text
"""


def test_splits_into_one_chunk_per_header_section():
    chunks = split_markdown(SYNTHETIC, source="synth.md")

    assert len(chunks) == 3


def test_headers_stay_in_the_chunk_text():
    chunks = split_markdown(SYNTHETIC, source="synth.md")

    starts = [chunk.text.split("\n", 1)[0] for chunk in chunks]
    assert starts == ["# Title", "## Section", "### Subsection"]


def test_heading_metadata_prefers_the_deepest_header_level():
    chunks = split_markdown(SYNTHETIC, source="synth.md")

    # The h3 section carries h1, h2 and h3 metadata — h3 must win, then h2.
    assert [chunk.heading for chunk in chunks] == ["Title", "Section", "Subsection"]


def test_source_is_stamped_on_every_chunk():
    chunks = split_markdown(SYNTHETIC, source="policy.md")

    assert all(chunk.source == "policy.md" for chunk in chunks)


def test_ids_increment_sequentially_from_start_id():
    chunks = split_markdown(SYNTHETIC, source="synth.md", start_id=5)

    assert [chunk.id for chunk in chunks] == [5, 6, 7]


def test_text_without_headers_becomes_a_single_chunk_with_no_heading():
    chunks = split_markdown("plain paragraph without any header", source="s.md")

    assert len(chunks) == 1
    assert chunks[0].heading is None
    assert chunks[0].text == "plain paragraph without any header"


def test_whitespace_only_text_yields_no_chunks():
    assert split_markdown("   \n\n  \n", source="s.md") == []


def test_oversized_section_falls_back_to_the_recursive_splitter():
    # One section of ~120 short space-separated words, far above chunk_size.
    long_md = "## Long section\n" + " ".join(f"word{i:03d}" for i in range(120))

    chunks = split_markdown(long_md, source="long.md", chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    # Words are short and space-separated, so the splitter can always honour
    # the limit exactly here.
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)
    # Every piece of the split section keeps the section's heading.
    assert all(chunk.heading == "Long section" for chunk in chunks)
    assert [chunk.id for chunk in chunks] == list(range(len(chunks)))
    # Nothing got lost at the edges of the split.
    joined = " ".join(chunk.text for chunk in chunks)
    assert "word000" in joined
    assert "word119" in joined


@pytest.mark.parametrize("path", CORPUS_PATHS, ids=lambda p: p.name)
def test_real_corpus_chunks_are_well_formed(path):
    chunks = split_markdown(path.read_text(encoding="utf-8"), source=path.name)

    assert chunks, "every corpus document must produce at least one chunk"
    assert [chunk.id for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.source == path.name
        assert chunk.text == chunk.text.strip()
        assert 0 < len(chunk.text) <= DEFAULT_CHUNK_SIZE
        # Every corpus file opens with a header, so no chunk is heading-less.
        assert chunk.heading is not None


def test_corpus_directory_holds_the_six_expected_documents():
    # The retrieval smoke tests depend on these files existing by name.
    assert [path.name for path in CORPUS_PATHS] == [
        "business_trips_policy.md",
        "employee_directory.md",
        "hardware_requests.md",
        "leave_policy.md",
        "salary_and_grades.md",
        "security_incidents.md",
    ]
