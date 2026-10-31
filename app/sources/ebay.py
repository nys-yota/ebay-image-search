from __future__ import annotations

import time
from typing import List, Optional

import httpx

from app.sources.base import Listing, SourceError

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
SCOPE = "https://api.ebay.com/oauth/api_scope"
MAX_PAGE_SIZE = 200  # eBay Browse API per-call cap


class EbaySource:
    """eBay Browse API keyword search (application OAuth token, no user login)."""

    name = "ebay"

    def __init__(
        self,
        client_id: Optional[str],
        client_secret: Optional[str],
        marketplace_id: str = "EBAY_US",
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._marketplace_id = marketplace_id
        self._client = client or httpx.AsyncClient(timeout=20)
        self._token: Optional[str] = None
        self._token_expires = 0.0

    async def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        if not self._client_id or not self._client_secret:
            raise SourceError("EBAY_CLIENT_ID / EBAY_CLIENT_SECRET are not set (see .env.example)")
        resp = await self._client.post(
            TOKEN_URL,
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials", "scope": SCOPE},
        )
        if resp.status_code != 200:
            raise SourceError(f"eBay token request failed: {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        self._token = body["access_token"]
        self._token_expires = time.time() + int(body.get("expires_in", 7200))
        return self._token

    async def search(self, query: str, limit: int) -> List[Listing]:
        token = await self._get_token()
        listings: List[Listing] = []
        offset = 0
        # Items without a picture are useless for image ranking, so we may need
        # more than one page to collect `limit` usable listings.
        while len(listings) < limit and offset < 1000:
            resp = await self._client.get(
                SEARCH_URL,
                params={"q": query, "limit": min(MAX_PAGE_SIZE, limit), "offset": offset},
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-EBAY-C-MARKETPLACE-ID": self._marketplace_id,
                },
            )
            if resp.status_code != 200:
                raise SourceError(f"eBay search failed: {resp.status_code} {resp.text[:200]}")
            items = resp.json().get("itemSummaries", [])
            if not items:
                break
            for item in items:
                listing = _parse_item(item)
                if listing:
                    listings.append(listing)
            offset += len(items)
        return listings[:limit]


def _parse_item(item: dict) -> Optional[Listing]:
    thumbs = item.get("thumbnailImages") or []
    image_url = (thumbs[0].get("imageUrl") if thumbs else None) or (item.get("image") or {}).get("imageUrl")
    url = item.get("itemWebUrl")
    if not image_url or not url:
        return None
    price = item.get("price") or {}
    return Listing(
        id=item.get("itemId", url),
        title=item.get("title", ""),
        url=url,
        image_url=image_url,
        price=price.get("value"),
        currency=price.get("currency"),
        source="ebay",
    )
