from __future__ import annotations

import threading
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image

from app.vocab import LABELS

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class Embedder(Protocol):
    model_id: str

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """L2-normalized embeddings, shape (n, d)."""
        ...

    def suggest_queries(self, image_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Top-k zero-shot text labels for an image embedding."""
        ...


class ClipEmbedder:
    """OpenCLIP ViT-B/32. The model is loaded lazily on first use."""

    model_id = f"{MODEL_NAME}/{PRETRAINED}"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"
        self._label_embeddings: Optional[np.ndarray] = None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import open_clip
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
            self._model = model.to(self._device).eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        import torch

        self._load()
        out: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [self._preprocess(img.convert("RGB")) for img in images[start : start + batch_size]]
            ).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype(np.float32))
        if not out:
            return np.zeros((0, 512), dtype=np.float32)
        return np.concatenate(out)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        import torch

        self._load()
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def suggest_queries(self, image_embedding: np.ndarray, k: int = 3) -> List[str]:
        if self._label_embeddings is None:
            self._label_embeddings = self._embed_texts([f"a photo of a {label}" for label in LABELS])
        scores = self._label_embeddings @ image_embedding
        return [LABELS[i] for i in np.argsort(-scores)[:k]]
