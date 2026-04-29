from pathlib import Path
from datetime import datetime
from src.scrapper import fetch_all_offers
from src.chat import build_filter_prompt, call_filter_ai
import json
import re

def parse_json_response(response: str) -> dict:
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        raise ValueError("Kein JSON gefunden!")
    return json.loads(match.group())

def load_or_fetch_offers():
    kw = datetime.now().isocalendar().week
    json_path = Path(f"data/json/KW{kw}_angebote_gefiltert.json")
    
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f), f"📂 Loaded cached deals from KW{kw}"
    else:
        offers_raw = fetch_all_offers()
        prompt = build_filter_prompt(offers_raw)
        response = call_filter_ai(prompt)
        offers = parse_json_response(response)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(offers, f, ensure_ascii=False, indent=2)
        return offers, f"🛒 Fetched fresh deals and saved to KW{kw}"
    
if __name__ == "__main__":
    offers, message = load_or_fetch_offers()
    print(message)
    print(json.dumps(offers, ensure_ascii=False, indent=2))