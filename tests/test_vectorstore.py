from clauseiq.embeddings import HashingEmbedder
from clauseiq.schemas import Chunk
from clauseiq.vectorstore import HybridVectorStore


def _store():
    store = HybridVectorStore(HashingEmbedder(dim=256))
    chunks = [
        Chunk(id="d::0", doc_id="d", ordinal=0,
              text="Either party may terminate on sixty days written notice."),
        Chunk(id="d::1", doc_id="d", ordinal=1,
              text="Invoices are payable within thirty days of the invoice date."),
        Chunk(id="d::2", doc_id="d", ordinal=2,
              text="Confidential information must be protected for five years."),
    ]
    store.add(chunks)
    return store


def test_search_ranks_relevant_first():
    store = _store()
    hits = store.search("how much notice to terminate the contract", top_k=2)
    assert hits
    assert "terminate" in hits[0].chunk.text.lower()


def test_hybrid_fuses_dense_and_sparse():
    store = _store()
    hits = store.search("payment invoice due", top_k=1, alpha=0.5)
    assert "invoice" in hits[0].chunk.text.lower()


def test_persistence_roundtrip(tmp_path):
    store = _store()
    p = tmp_path / "idx.pkl"
    store.save(str(p))
    store2 = HybridVectorStore(HashingEmbedder(dim=256))
    store2.load(str(p))
    assert store2.stats()["chunks"] == 3
    assert store2.search("terminate notice", top_k=1)
