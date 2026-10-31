from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class Listing:
    id: str
    title: str
    url: str
    image_url: str
    price: Optional[str] = None
    currency: Optional[str] = None
    source: str = ""


class SourceError(Exception):
    """Raised when a listing source is misconfigured or its API call fails."""


class ListingSource(Protocol):
    async def search(self, query: str, limit: int) -> List[Listing]:
        """Return up to `limit` listings (with an image) matching a keyword query."""
        ...
