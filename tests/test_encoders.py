import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rageval.serve import encoders as enc


@pytest.fixture
def fake_models(monkeypatch):
    class FakeEmb:
        def __init__(self, *a, **k):
            pass

        def embed(self, texts):
            return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    class FakeCE:
        def __init__(self, *a, **k):
            pass

        def score(self, pairs):
            return np.array([float(len(d)) for _, d in pairs], dtype=np.float32)

    import rageval.models as models

    monkeypatch.setattr(models, "HFEmbedder", FakeEmb)
    monkeypatch.setattr(models, "HFCrossEncoder", FakeCE)


def test_embed_and_rerank_servers(fake_models):
    with TestClient(enc.create_encoder_app("embed", "fake")) as c:
        r = c.post("/v1/embeddings", json={"input": ["a", "bbb"]}).json()
        assert [d["embedding"][0] for d in r["data"]] == [1.0, 3.0]
        assert c.post("/rerank", json={"query": "q", "documents": ["x"]}).status_code == 404
    with TestClient(enc.create_encoder_app("rerank", "fake")) as c:
        r = c.post("/rerank", json={"query": "q", "documents": ["xx", "x"]}).json()
        assert [x["relevance_score"] for x in r["results"]] == [2.0, 1.0]
        assert c.get("/stats").json()["items"] == 2


async def test_remote_clients_against_servers(fake_models):
    app = enc.create_encoder_app("rerank", "fake")
    with TestClient(app):  # runs the lifespan (loads the fake model)
        client = enc.RemoteRerankClient("http://enc")
        client._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://enc"
        )
        s = await client.submit([("q", "abc"), ("q", "a"), ("p", "ab")])
        assert s.tolist() == [3.0, 1.0, 2.0]
