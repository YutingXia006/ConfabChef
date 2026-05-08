from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, Literal, List, Any, Dict
from src.retriever import load_retriever
from src.offers import load_or_fetch_offers
import operator
import streamlit as st

# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    offers_section: str
    route: str


def build_agent():
    # ── Supermärkte ──────────────────────────────────────────────────────────────
    SUPPORTED_MARKETS = ['Lidl', 
                         'REWE', 
                         'Penny',
                         'Netto Marken-Discount',
                         'Action', 
                         'Hornbach', 
                         'Müller', 
                         'EDEKA', 
                         'Penny', 
                         'Globus-Baumarkt', 
                         'E center', 
                         'Marktkauf', 
                         'Netto Marken-Discount', 
                         'OBI', 
                         'Müller', 
                         'Center Parcs', 
                         'Netto Marken-Discount', 
                         'MediaMarkt Saturn', 
                         'dm-drogerie markt', 
                         'Dehner Garten-Center']
    # ── LLM & Retriever ──────────────────────────────────────────────────────────
    llm_generate = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=8000,
        temperature=0.7
    )
    llm_router = ChatGroq(
        model="llama-3.1-8b-instant", 
        max_tokens=10,  # braucht nur ein Wort zurückgeben
        temperature=0.0  # kein Zufall beim Routing
    )
    llm_extractor = ChatGroq(
        model="llama-3.1-8b-instant",
        max_tokens=100,
        temperature=0.0
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
        
        messages = prompt.format_messages(
            context=context,
            offers_section=offers_section,
            question=question
        )
        response = llm_generate.invoke(messages)
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

    workflow.add_edge("get_offers", "generate")
    workflow.add_edge("meal_plan_offers", "generate")
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
            "route": ""
        })
        print(f"Route: {result['route']}")
        print(f"Markets: {st.session_state.get('selected_markets', 'default')}")