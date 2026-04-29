from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.retriever import load_retriever
import warnings
import logging
import streamlit as st
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

load_dotenv(Path(__file__).parent.parent / ".env")

llm_filter = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens=8000,
    temperature=0.7
)

def call_filter_ai(prompt: str) -> str:
    response = llm_filter.invoke([HumanMessage(content=prompt)])
    return str(response.content)

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

def build_chain():
    retriever = load_retriever()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=8000,
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template("""
    You are ConfabChef (CC), a friendly AI cooking assistant 
    specializing in international cuisine.

    Use the following recipes from your knowledge base:
    {context}

    User preferences (if mentioned): apply dietary restrictions, 
    calorie goals, and allergies from the conversation.

    When creating meal plans:
    - Write a table for the overview
    - Structure by day with Breakfast (25%), Lunch (40%), Dinner (35%)
    - Include ingredients with exact amounts
    - Include macros per serving (Calories, Protein, Carbs, Fat, Fiber)
    - Include step-by-step instructions
    - Add tips and substitutions
    - End with a shopping list sorted by category

    Answer in the same language as the user.

    User question: {question}
    """)

    def chain(question: str) -> str:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Angebote als extra Kontext hinzufügen
        offers_context = ""
        if "offers" in st.session_state:
            offers_context = f"\n\nCurrent supermarket deals:\n"
            for market, items in st.session_state.offers.items():
                offers_context += f"\n{market}:\n"
                for item in items:
                    offers_context += f"- {item['name']} ({item['price_eur']}€)\n"
        
        messages = prompt.format_messages(
            context=context + offers_context,
            question=question
        )
        response = llm.invoke(messages)
        return str(response.content)

    return chain

if __name__ == "__main__":
    chain = build_chain()
    print(chain("Can you suggest a healthy Chinese dinner?"))