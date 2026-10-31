import io
from typing import List

import httpx
import numpy as np
import pytest
from PIL import Image

from app.main import create_app
from app.sources.base import Listing, SourceError


def png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeEmbedder:
    """Embeds an image as its (normalized) mean RGB, so similar colors are similar."""

    model_id = "fake"

    def __init__(self):
        self.embedded = 0

    def embed_images(self, images) -> np.ndarray:
        self.embedded += len(images)
        vecs = np.array([np.asarray(i.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) for i in images])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def suggest_queries(self, image_embedding, k=3) -> List[str]:
        return ["red thing", "blue thing", "green thing"][:k]


class FakeSource:
    def __init__(self, listings):
        self.listings = listings
        self.queries = []

    async def search(self, query, limit):
        self.queries.append(query)
        return self.listings[:limit]


COLORS = {"red": (250, 10, 10), "blue": (10, 10, 250), "orange": (250, 120, 10)}
LISTINGS = [Listing(id=n, title=f"{n} item", url=f"https://x/{n}", image_url=f"https://img/{n}.png", source="fake") for n in COLORS]


def img_handler(request: httpx.Request) -> httpx.Response:
    name = request.url.path.rsplit("/", 1)[-1].removesuffix(".png")
    if name in COLORS:
        return httpx.Response(200, content=png(COLORS[name]))
    return httpx.Response(404)


@pytest.fixture
def setup(tmp_path):
    from fastapi.testclient import TestClient

    embedder, source = FakeEmbedder(), FakeSource(list(LISTINGS))
    http = httpx.AsyncClient(transport=httpx.MockTransport(img_handler))
    app = create_app(embedder=embedder, source=source, http_client=http, cache_dir=tmp_path)
    return TestClient(app), embedder, source


def post(client, color=(255, 0, 0), **form):
    return client.post("/api/search", files={"image": ("q.png", png(color), "image/png")}, data=form)


def test_results_are_ranked_by_visual_similarity(setup):
    client, _, source = setup
    body = post(client, (255, 0, 0)).json()
    assert [r["id"] for r in body["results"]] == ["red", "orange", "blue"]
    assert body["results"][0]["score"] > body["results"][1]["score"] > body["results"][2]["score"]
    assert body["query"] == "red thing"  # auto-suggested
    assert source.queries == ["red thing"]


def test_explicit_query_overrides_suggestion_and_limit_applies(setup):
    client, _, source = setup
    body = post(client, query="floor lamp", limit="1").json()
    assert source.queries == ["floor lamp"]
    assert body["query"] == "floor lamp" and len(body["results"]) == 1


def test_listing_embeddings_are_cached_between_searches(setup):
    client, embedder, _ = setup
    post(client)
    after_first = embedder.embedded
    assert after_first == 1 + len(LISTINGS)  # query + each listing
    post(client)
    assert embedder.embedded == after_first + 1  # only the query image again


def test_listings_with_broken_images_are_dropped(setup):
    client, _, source = setup
    source.listings.append(Listing(id="bad", title="bad", url="https://x/bad", image_url="https://img/missing.png"))
    body = post(client).json()
    assert "bad" not in [r["id"] for r in body["results"]]
    assert body["candidates_considered"] == len(LISTINGS)


def test_non_image_upload_is_400(setup):
    client, _, _ = setup
    r = client.post("/api/search", files={"image": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_source_error_is_503(tmp_path):
    from fastapi.testclient import TestClient

    class Broken:
        async def search(self, query, limit):
            raise SourceError("no keys")

    app = create_app(embedder=FakeEmbedder(), source=Broken(), http_client=httpx.AsyncClient(), cache_dir=tmp_path)
    r = post(TestClient(app))
    assert r.status_code == 503 and "no keys" in r.json()["detail"]


def test_index_page_served(setup):
    client, _, _ = setup
    assert client.get("/").status_code == 200
