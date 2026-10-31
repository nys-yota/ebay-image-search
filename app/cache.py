from __future__ import annotations

import asyncio
import hashlib
import io
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from PIL import Image


class EmbeddingCache:
    """On-disk cache of image embeddings, keyed by (model id, image URL)."""

    def __init__(self, root: Path, model_id: str):
        self._dir = Path(root) / hashlib.sha1(model_id.encode()).hexdigest()[:12]
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        return self._dir / (hashlib.sha1(url.encode()).hexdigest() + ".npy")

    def get(self, url: str) -> Optional[np.ndarray]:
        path = self._path(url)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None

    def put(self, url: str, vec: np.ndarray) -> None:
        np.save(self._path(url), vec)


async def download_images(
    client: httpx.AsyncClient, urls: List[str], concurrency: int = 16
) -> Dict[str, Image.Image]:
    """Fetch and decode images concurrently. URLs that fail are simply omitted."""
    sem = asyncio.Semaphore(concurrency)

    async def fetch(url: str):
        async with sem:
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                img.load()
                return url, img
            except (httpx.HTTPError, OSError, ValueError):
                return url, None

    results = await asyncio.gather(*(fetch(u) for u in urls))
    return {url: img for url, img in results if img is not None}
