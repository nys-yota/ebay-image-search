from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.cache import EmbeddingCache, download_images
from app.embeddings import ClipEmbedder, Embedder
from app.ranking import rank_by_similarity
from app.sources.base import ListingSource, SourceError
from app.sources.ebay import EbaySource

STATIC_DIR = Path(__file__).parent / "static"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def create_app(
    embedder: Optional[Embedder] = None,
    source: Optional[ListingSource] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    cache_dir: Optional[Path] = None,
) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Marketplace image search")
    embedder = embedder or ClipEmbedder()
    http_client = http_client or httpx.AsyncClient(timeout=20)
    source = source or EbaySource(
        os.getenv("EBAY_CLIENT_ID"),
        os.getenv("EBAY_CLIENT_SECRET"),
        marketplace_id=os.getenv("EBAY_MARKETPLACE_ID", "EBAY_US"),
    )
    emb_cache = EmbeddingCache(cache_dir or CACHE_DIR, embedder.model_id)

    @app.post("/api/search")
    async def search(
        image: UploadFile = File(...),
        query: Optional[str] = Form(None),
        limit: int = Form(24),
        candidates: int = Form(100),
    ):
        limit = max(1, min(limit, 100))
        candidates = max(limit, min(candidates, 200))

        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image too large (max 15 MB)")
        try:
            query_img = Image.open(io.BytesIO(data))
            query_img.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Could not read the uploaded file as an image")

        query_vec = (await run_in_threadpool(embedder.embed_images, [query_img]))[0]
        suggestions: List[str] = await run_in_threadpool(embedder.suggest_queries, query_vec, 3)
        query_text = (query or "").strip() or suggestions[0]

        try:
            listings = await source.search(query_text, candidates)
        except SourceError as e:
            raise HTTPException(503, str(e))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Listing source unreachable: {e}")

        vectors = await _embed_listings(listings, embedder, emb_cache, http_client)
        usable = [l for l in listings if l.image_url in vectors]
        ranked = rank_by_similarity(query_vec, np.stack([vectors[l.image_url] for l in usable]), limit) if usable else []

        return {
            "query": query_text,
            "suggested_queries": suggestions,
            "candidates_considered": len(usable),
            "results": [
                {
                    "id": usable[i].id,
                    "title": usable[i].title,
                    "price": usable[i].price,
                    "currency": usable[i].currency,
                    "url": usable[i].url,
                    "image_url": usable[i].image_url,
                    "source": usable[i].source,
                    "score": round(score, 4),
                }
                for i, score in ranked
            ],
        }

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _embed_listings(listings, embedder: Embedder, cache: EmbeddingCache, client: httpx.AsyncClient) -> Dict[str, np.ndarray]:
    """Embeddings for each listing image URL, using the disk cache where possible."""
    vectors: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for url in dict.fromkeys(l.image_url for l in listings):
        cached = cache.get(url)
        if cached is not None:
            vectors[url] = cached
        else:
            missing.append(url)
    if missing:
        images = await download_images(client, missing)
        urls = list(images)
        if urls:
            embedded = await run_in_threadpool(embedder.embed_images, [images[u] for u in urls])
            for url, vec in zip(urls, embedded):
                cache.put(url, vec)
                vectors[url] = vec
    return vectors


app = create_app()
