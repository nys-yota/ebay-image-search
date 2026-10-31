import httpx
import pytest

from app.sources.base import SourceError
from app.sources.ebay import SEARCH_URL, TOKEN_URL, EbaySource

ITEMS = {
    "itemSummaries": [
        {
            "itemId": "v1|1|0",
            "title": "Mid-century floor lamp",
            "price": {"value": "45.00", "currency": "USD"},
            "itemWebUrl": "https://www.ebay.com/itm/1",
            "image": {"imageUrl": "https://i.ebayimg.com/1-big.jpg"},
            "thumbnailImages": [{"imageUrl": "https://i.ebayimg.com/1-thumb.jpg"}],
        },
        {  # no image: must be skipped
            "itemId": "v1|2|0",
            "title": "No picture",
            "price": {"value": "5.00", "currency": "USD"},
            "itemWebUrl": "https://www.ebay.com/itm/2",
        },
        {
            "itemId": "v1|3|0",
            "title": "Brass lamp",
            "itemWebUrl": "https://www.ebay.com/itm/3",
            "image": {"imageUrl": "https://i.ebayimg.com/3.jpg"},
        },
    ]
}


def make_source(search_handler, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 7200})
        assert str(request.url).startswith(SEARCH_URL)
        return search_handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EbaySource("id", "secret", client=client)


async def test_search_parses_listings_and_skips_imageless():
    calls = []
    src = make_source(lambda r: httpx.Response(200, json=ITEMS) if r.url.params["offset"] == "0" else httpx.Response(200, json={}), calls)
    listings = await src.search("lamp", 10)
    assert [l.id for l in listings] == ["v1|1|0", "v1|3|0"]
    assert listings[0].image_url.endswith("1-thumb.jpg")  # prefers thumbnail
    assert listings[0].price == "45.00" and listings[0].currency == "USD"
    assert listings[1].price is None
    assert any(c.headers.get("authorization") == "Bearer tok" for c in calls[1:])


async def test_token_is_reused_across_searches():
    calls = []
    src = make_source(lambda r: httpx.Response(200, json={}), calls)
    await src.search("a", 5)
    await src.search("b", 5)
    assert sum(str(c.url) == TOKEN_URL for c in calls) == 1


async def test_missing_credentials_raises_source_error():
    with pytest.raises(SourceError, match="EBAY_CLIENT_ID"):
        await EbaySource(None, None).search("lamp", 5)


async def test_search_error_status_raises_source_error():
    src = make_source(lambda r: httpx.Response(500, text="boom"), [])
    with pytest.raises(SourceError, match="500"):
        await src.search("lamp", 5)
