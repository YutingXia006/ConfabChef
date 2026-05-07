from pathlib import Path
from datetime import datetime
from src.scraper import fetch_all_offers
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import json
import re
from dotenv import load_dotenv
load_dotenv()

llm_filter = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens=8000,
    temperature=0.7
)

def call_filter_ai(prompt: str) -> str:
    response = llm_filter.invoke([HumanMessage(content=prompt)])
    return str(response.content)

def parse_json_response(response: str) -> dict:
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        raise ValueError("Kein JSON gefunden!")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Ungültiges JSON: {e}")

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
    
def format_angebote(angebote: dict, preise: bool) -> str:
    lines = []
    if preise:
        for markt, items in angebote.items():
            lines.append(f"\n{markt}:")
            for item in items:
                lines.append(f"  - {item['name']}: {item['price_eur']}€")
        return "\n".join(lines)
    else:
        for markt, items in angebote.items():
            lines.append(f"\n{markt}:")
            for item in items:
                lines.append(f"  - {item['name']}")
        return "\n".join(lines)

def build_filter_prompt(angebote: dict) -> str:
    return f"""
    Du bekommst eine Liste von Supermarktangeboten.
    Extrahiere NUR Produkte die zum Kochen geeignet sind.

    **Ignoriere:**
    - Alkohol (Bier, Wein, Spirituosen)
    - Süßigkeiten & Snacks (Chips, Schokolade, Kekse, Eis)
    - Non-Food Artikel (Holzkohle, Pfannen, etc.)
    - Fertiggerichte & Fast Food
    - Softdrinks & Energy Drinks

    **Behalte:**
    - Fleisch, Fisch, Meeresfrüchte
    - Gemüse & Obst
    - Milchprodukte & Käse
    - Nudeln, Reis, Getreide
    - Saucen, Gewürze, Öle
    - Brot & Backwaren (zum Frühstück)
    - Säfte & Wasser

    **Angebote:**
    {format_angebote(angebote, True)}

    **Ausgabe als JSON:**
    {{
    "Lidl": [
        {{"name": "Produktname", "price_eur": 1.99}}
    ],
    "EDEKA": [...]
    }}

    Keine Erklärungen, nur JSON.
    """
    
if __name__ == "__main__":
    offers, message = load_or_fetch_offers()
    print(message)
    print(json.dumps(offers, ensure_ascii=False, indent=2))