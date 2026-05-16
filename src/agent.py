from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, List, Any, Dict
from src.retriever import load_retriever
from src.offers import load_or_fetch_offers
import operator
import streamlit as st
import random

# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    offers_section: str
    route: str
    draft_plan: str


def build_agent():
    # ── Supermärkte ──────────────────────────────────────────────────────────────
    SUPPORTED_MARKETS = [
        "Lidl", "REWE", "EDEKA", "Penny", 
        "Netto Marken-Discount", "E center", "Marktkauf"
    ]
    # ── LLM & Retriever ──────────────────────────────────────────────────────────
    llm_generate = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        temperature=0.7
    )
    llm_router = ChatGroq(
        model="llama-3.1-8b-instant", 
        max_tokens=10,  # braucht nur ein Wort zurückgeben
        temperature=0.0  # kein Zufall beim Routing
    )
    llm_planner = ChatGroq(
    model="llama-3.1-8b-instant",  # Planner braucht kein 70B
    max_tokens=500,
    temperature=0.7
    )
    llm_dietitian = ChatGroq(
        model="llama-3.3-70b-versatile",  # Nur Dietitian braucht 70B
        max_tokens=3000,
        temperature=0.3
    )
    retriever = load_retriever()

    # ── Nodes ────────────────────────────────────────────────────────────────────
    def router_node(state: AgentState) -> Dict[str, Any]:
        """LLM entscheidet welchen Weg der Agent nimmt"""
        question = state["messages"][-1].content
        
        system = """You are a routing assistant. Classify the user question into exactly one category:
        - "recipes": user wants a specific recipe or ingredient-based suggestion
        - "meal_plan": user wants a meal plan (day/week)
        - "offers": user asks what's on sale or wants deal-based suggestions
        - "general": cooking tips, questions, anything else
        
        Respond with ONLY the category word, nothing else."""
        
        response = llm_router.invoke([
            SystemMessage(content=system),
            HumanMessage(content=question)
        ])
        
        route = str(response.content).strip().lower()
        if route not in ["recipes", "meal_plan", "offers", "general"]:
            route = "general"
        
        return {"route": route, "messages": []}

    def search_recipes_node(state: AgentState) ->  Dict[str, Any]:
        """RAG Retrieval"""
        question = str(state["messages"][-1].content)
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context, "messages": []}

    def get_offers_node(state: AgentState) -> Dict[str, Any]:
        markets = st.session_state.get("selected_markets")
        cached_markets = st.session_state.get("loaded_markets")

        # Neu laden wenn Märkte sich geändert haben
        if not st.session_state.get("offers") or cached_markets != markets:
            offers, _ = load_or_fetch_offers(markets=markets)
            st.session_state.offers = offers
            st.session_state.loaded_markets = markets  # merken welche Märkte geladen wurden

        offers = st.session_state.offers

        offers_text = "Current supermarket deals (prefer but not exclusively):\n"
        for market, items in offers.items():
            offers_text += f"\n{market}:\n"
            for item in items:
                offers_text += f"- {item['name']} ({item['price_eur']}€)\n"

        return {"offers_section": offers_text, "messages": []}
    
    def extract_market_node(state: AgentState) -> Dict[str, Any]:
        question = str(state["messages"][-1].content).lower()
        
        found = [m for m in SUPPORTED_MARKETS if m.lower() in question]
        
        if found:
            st.session_state.selected_markets = found
        
        return {"messages": []}

    def generate_node(state: AgentState) -> Dict[str, Any]:
        question = str(state["messages"][-1].content)
        context = state.get("context", "")
        offers_section = state.get("offers_section", "")
        
        prompt = ChatPromptTemplate.from_template("""
        You are ConfabChef (CC), a friendly AI cooking assistant 
        specializing in international cuisine.

        Use the following recipes from your knowledge base:
        {context}

        {offers_section}

        Apply dietary restrictions, calorie goals, and allergies if mentioned.

        For recipes: include ingredients with amounts and step-by-step instructions.
        For offers/suggestions: give a brief overview with 2-3 recipe ideas.

        Answer in the same language as the user.

        User question: {question}
        """)
        
        messages = prompt.format_messages(
            context=context,
            offers_section=offers_section,
            question=question
        )
        response = llm_generate.invoke(messages)
        return {"messages": [AIMessage(content=str(response.content))]}
    
    def planner_node(state: AgentState) -> Dict[str, Any]:
        question = str(state["messages"][-1].content)
        context = state.get("context", "")
        
        # Zufällige Küchen für mehr Abwechslung
        cuisines = ["Italian", "Japanese", "Mexican", "Indian", "Thai", 
                    "Greek", "Korean", "Lebanese", "Peruvian", "Ethiopian"]
        random.shuffle(cuisines)
        cuisine_suggestion = ", ".join(cuisines[:4])
        
        prompt = ChatPromptTemplate.from_template("""
        Create ONLY dish names for a weekly meal plan.
        No ingredients, no instructions, no macros. Just names.
        
        Try to incorporate these cuisines for variety: {cuisines}
        Use these recipes as inspiration: {context}
        
        Format: Day — Breakfast / Lunch / Dinner
        Maximum 100 words total.
        
        User request: {question}
        """)
        
        messages = prompt.format_messages(
            context=context,
            question=question,
            cuisines=cuisine_suggestion
        )
        response = llm_planner.invoke(messages)
        return {"draft_plan": str(response.content), "messages": []}

    def dietitian_node(state: AgentState) -> Dict[str, Any]:
        """Überarbeitet Plan mit Einschränkungen und Angeboten"""
        question = str(state["messages"][-1].content)
        draft = state.get("draft_plan", "")
        offers_section = state.get("offers_section", "")
        
        prompt = ChatPromptTemplate.from_template("""
        You are a dietitian reviewing a meal plan draft.
        
        Original draft:
        {draft}
        
        {offers_section}
        
        Your job:
        1. Apply ALL dietary restrictions from the user request
        2. Substitute forbidden ingredients with the MOST DIFFERENT valid alternative, not just the closest one (e.g. pork → lentils, not pork → beef)
        3. Prefer ingredients from supermarket deals where possible
        4. NEVER repeat the same protein source on the same day (e.g. if breakfast has eggs, lunch and dinner cannot)
        5. Add exact amounts, macros, instructions and shopping list
        
        Present the final plan as:
        1. A markdown overview table: | Day | Breakfast | Lunch | Dinner |
        2. A shopping list sorted by category with supermarket where available
        
        Do NOT include:
        - Individual recipes or cooking instructions
        - Detailed macros per meal
        - Notes or disclaimers

        The user can ask for specific recipes or macros separately.

        User request with restrictions: {question}
        """)
        
        messages = prompt.format_messages(
            draft=draft,
            offers_section=offers_section,
            question=question
        )
        response = llm_dietitian.invoke(messages)
        return {"messages": [AIMessage(content=str(response.content))]}

    def after_recipes(state: AgentState) -> str:
        """Nach recipe search — wohin?"""
        if state["route"] == "meal_plan":
            return "meal_plan_offers"
        return "generate"

    def where_to_go(state: AgentState) -> str:
        """
        Diese Funktion liest einfach den im router_node gesetzten Pfad aus 
        und sagt dem Graphen, wo er als nächstes hin soll.
        """
        return state["route"]
    
    def after_extract(state: AgentState) -> str:
        if state["route"] == "offers":
            return "get_offers"
        return "search_recipes"

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("search_recipes", search_recipes_node)
    workflow.add_node("get_offers", get_offers_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("meal_plan_offers", get_offers_node)
    workflow.add_node("extract_market", extract_market_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("dietitian", dietitian_node)

    workflow.add_edge(START, "router")

    # Router → extract_market → search_recipes oder get_offers
    workflow.add_conditional_edges("router", where_to_go, {
        "recipes": "search_recipes",
        "offers": "extract_market",
        "meal_plan": "extract_market",  # immer durch market extraction
        "general": END
    })

    workflow.add_conditional_edges("extract_market", after_extract, {
        "search_recipes": "search_recipes",
        "get_offers": "get_offers"
    })

    # search_recipes entscheidet selbst wohin
    workflow.add_conditional_edges("search_recipes", after_recipes, {
        "meal_plan_offers": "meal_plan_offers",
        "generate": "generate"
    })
    # meal_plan Flow:
    # search_recipes → meal_plan_offers → planner → dietitian
    workflow.add_edge("meal_plan_offers", "planner")
    workflow.add_edge("planner", "dietitian")
    workflow.add_edge("dietitian", END)

    workflow.add_edge("get_offers", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / ".env")
    
    agent = build_agent()
    
    tests = [
        "I shop at Lidl, give me a weekly meal plan",
        "Give me a chicken recipe",
        "What's on sale this week at EDEKA?",
        "How do I julienne carrots?",
    ]
    
    for question in tests:
        print(f"\nQ: {question}")
        result = agent.invoke({
            "messages": [HumanMessage(content=question)],
            "context": "",
            "offers_section": "",
            "route": "",
            "draft_plan": ""
        })
        print(f"Route: {result['route']}")
        print(f"Markets: {st.session_state.get('selected_markets', 'default')}")