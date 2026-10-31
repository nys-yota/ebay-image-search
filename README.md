# ebay-image-search

Upload a photo of an item and get back marketplace listings that look like it.

Marketplaces don't offer search-by-image over their whole catalog, so this works in two steps:

1. A keyword search fetches candidate listings. The keywords come from you, or from CLIP zero-shot labelling of your photo.
2. Each candidate's thumbnail is embedded with [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B/32) and the candidates are re-ranked by cosine similarity to your photo.

Listing sources are pluggable (`app/sources/base.py`). The first one is the eBay Browse API. Facebook Marketplace has no public API and scraping it violates Meta's Terms of Service, so it is not included.

## Setup

Python 3.9+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # then add your eBay keys from https://developer.ebay.com
```

## Run

```sh
.venv/bin/uvicorn app.main:app --reload
```

Open http://localhost:8000. The first search downloads the CLIP weights (about 600 MB) and is slow. Later searches reuse them. Listing embeddings are cached in `.cache/`, so repeated searches are faster.

## Test

```sh
.venv/bin/pytest
```

## Adding a source

Implement `async def search(query, limit) -> list[Listing]` (see `app/sources/base.py`) and pass the instance to `create_app(source=...)`.
