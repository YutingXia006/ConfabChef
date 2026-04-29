import re
import requests
import os
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(Path(__file__).parent.parent / ".env")

try:
    RELEVANT_SUPERMARKTS = os.environ["MARKETS"].split(",")
except (TypeError, ValueError):
    raise ValueError("MARKETS must be set in your .env file")

try:
    LAT = float(os.environ["LAT"])
    LNG = float(os.environ["LNG"])
except (TypeError, ValueError):
    raise ValueError("LATITUDE and LONGITUDE must be set in your .env file")

def fetch_brochure_ids():
    html = requests.get(
        "https://www.kaufda.de/shelf",
        params={"lat": LAT, "lng": LNG}
    ).text

    # Händlername + ID zusammen extrahieren
    matches = re.findall(
        r'font-bold[^>]*>([^<]+)</div>.*?bonial\.biz/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
        html
    )

    brochure_ids = {} 

    for name, id in matches:
        if name in RELEVANT_SUPERMARKTS and name not in brochure_ids:
            brochure_ids[name] = id
    return brochure_ids

def fetch_brochure_pages(brochure_id: str, lat: float, lng: float) -> dict:
    url = f"https://content-viewer-be.kaufda.de/v1/brochures/{brochure_id}/pages"
    params = {
        "partner": "kaufda_web",
        "brochureKey": "",
        "lat": lat,
        "lng": lng
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def extract_price(deals: list) -> float | None:
    for deal in deals:
        if deal.get("type") == "SALES_PRICE":
            return deal.get("min")
    return None


def extract_categories(category_paths: list) -> dict:
    names = [cp["name"] for cp in category_paths]
    return {
        "main": names[0] if names else None,
        "sub": names[-1] if names else None,
        "path": " > ".join(names)
    }

def is_food(category: dict) -> bool:
    return category.get("main") == "Lebensmittel und Getränke"

def parse_offer(offer_obj: dict) -> list[dict]:
    content = offer_obj.get("content", {})
    price = extract_price(content.get("deals", []))

    result = []
    for product in content.get("products", []):
        category = extract_categories(product.get("categoryPaths", []))

        if not is_food(category):
            continue

        name = product.get("name", "")
        brand = product.get("brandName", "")

        result.append({
            "name": f"{brand} {name}".strip(),
            "price_eur": price,
            "category": category["sub"]
        })
    return result

def parse_food_offers(data: dict) -> list[dict]:
    angebote = []
    for page in data.get("contents", []):
        for offer_obj in page.get("offers", []):
            angebote.extend(parse_offer(offer_obj))
    return angebote

def fetch_all_offers():
    brochure_ids = fetch_brochure_ids()
    all_offers = {}
    for name, brochure_id in brochure_ids.items():
        raw_data = fetch_brochure_pages(brochure_id, LAT, LNG)
        angebote = parse_food_offers(raw_data)
        all_offers[name] = angebote
    return all_offers

if __name__ == "__main__":
    print(fetch_all_offers())