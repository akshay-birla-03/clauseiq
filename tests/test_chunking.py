from clauseiq.chunking import chunk_document, split_into_clauses

SAMPLE = """1. DEFINITIONS
1.1 "Services" means the analytics services.

2. TERM
2.1 This Agreement continues for 24 months.

3. TERMINATION
3.1 Either party may terminate on 60 days notice.
"""


def test_split_finds_clauses():
    clauses = split_into_clauses(SAMPLE)
    assert len(clauses) >= 3
    assert any("TERMINATION" in c for c in clauses)


def test_chunk_ids_unique_and_ordered():
    chunks = chunk_document("doc1", SAMPLE, chunk_size=40, overlap=8)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.ordinal for c in chunks] == sorted(c.ordinal for c in chunks)
    assert all(c.doc_id == "doc1" for c in chunks)


def test_chunk_preserves_content():
    chunks = chunk_document("doc1", SAMPLE, chunk_size=1000, overlap=0)
    joined = " ".join(c.text for c in chunks)
    assert "60 days" in joined
