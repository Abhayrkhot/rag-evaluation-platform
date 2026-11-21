import numpy as np

from rageval.retrieval.bm25 import BM25Index
from rageval.retrieval.dense import DenseIndex, format_doc, format_query

DOCS = {
    "d1": "Scott Derrickson\nScott Derrickson is an American director, screenwriter and producer.",
    "d2": "Ed Wood\nEdward Davis Wood Jr. was an American filmmaker, actor, writer, producer, and director.",
    "d3": "Paris\nParis is the capital and most populous city of France.",
}


def test_bm25_ranks_lexical_match_first(tmp_path):
    idx = BM25Index().build(list(DOCS), list(DOCS.values()))
    hits = idx.search(["capital of France", "Ed Wood filmmaker"], k=2)
    assert hits[0][0].doc_id == "d3"
    assert hits[1][0].doc_id == "d2"
    idx.save(tmp_path / "bm25")
    again = BM25Index.load(tmp_path / "bm25")
    assert [h.doc_id for h in again.search(["capital of France"], k=1)[0]] == ["d3"]


def test_bm25_unknown_terms_return_nothing():
    idx = BM25Index().build(list(DOCS), list(DOCS.values()))
    assert idx.search(["zzzzqqq"], k=3) == [[]]


def test_dense_flat_and_hnsw_agree(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(500, 32)).astype(np.float32)
    ids = [f"d{i}" for i in range(500)]
    flat = DenseIndex("flat").build(ids, vecs)
    hnsw = DenseIndex("hnsw", m=16).build(ids, vecs)
    q = vecs[:5] + 0.01 * rng.normal(size=(5, 32)).astype(np.float32)
    f = flat.search(q, k=1)
    h = hnsw.search(q, k=1, ef_search=64)
    assert [r[0].doc_id for r in f] == ids[:5]
    assert [r[0].doc_id for r in h] == ids[:5]
    hnsw.save(tmp_path / "h")
    assert DenseIndex.load(tmp_path / "h").search(q, k=1)[0][0].doc_id == "d0"


def test_query_and_doc_formats():
    assert format_query("Qwen/Qwen3-Embedding-0.6B", "who?").startswith("Instruct: ")
    assert format_query("BAAI/bge-base-en-v1.5", "who?").startswith("Represent this sentence")
    assert format_query("unknown/model", "who?") == "who?"
    assert format_doc("intfloat/e5-base-v2", "T", "x") == "passage: T\nx"
