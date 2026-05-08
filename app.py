from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage
from groq import RateLimitError
import re
from src.ingest import load_csv_recipes, load_txt_recipes, split_documents, build_faiss_index, FAISS_INDEX
from src.agent import build_agent
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv(Path(__file__).parent / ".env")
@st.cache_resource
def get_agent():
    return build_agent()

st.set_page_config(
    page_title="ConfabChef",
    page_icon="🍳",
    layout="centered"
)

st.title("🍳 ConfabChef (CC)")
st.caption("Your AI sous-chef — because even AI chefs occasionally confabulate")

if not FAISS_INDEX.exists():
    with st.status("📚 Building recipe database for the first time...", expanded=True) as status:
        st.write("Loading recipes...")
        csv_docs = load_csv_recipes()
        txt_docs = load_txt_recipes()
        st.write(f"Found {len(csv_docs + txt_docs)} recipes!")
        st.write("Building search index (this may take a few minutes)...")
        chunks = split_documents(txt_docs)
        all_docs = csv_docs + chunks
        build_faiss_index(all_docs)
        status.update(label="✅ Recipe database ready!", state="complete")

if "agent" not in st.session_state:
    with st.status("🍳 Preparing the kitchen...", expanded=True) as status:
        st.write("Loading recipe database...")
        st.write("Initializing AI model...")
        st.session_state.agent = build_agent()
        status.update(label="✅ ConfabChef is ready!", state="complete")

if st.session_state.get("offers"):
    markets = ", ".join(st.session_state.offers.keys())
    st.caption(f"🛒 Deals loaded: {markets}")

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome = """👋 Hi! I'm **ConfabChef (CC)**, your AI sous-chef!

Here's what I can do:
- 🍳 **Recipes** — "Give me a quick chicken recipe"
- 📅 **Meal plans** — "Plan my week with high protein meals"  
- 🛒 **Deals** — "What can I cook with this week's offers?"

💡 **Tip:** Tell me your supermarket and I'll use their current deals automatically!
> *"I usually shop at Lidl, give me a weekly meal plan"*"""
    
    st.session_state.messages.append({"role": "assistant", "content": welcome})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me for a recipe or meal plan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        try:
            with st.status("🍳 Cooking...", expanded=True) as status:
                st.write("🔍 Analyzing your request...")
                result = st.session_state.agent.invoke({
                    "messages": [HumanMessage(content=prompt)],
                    "context": "",
                    "offers_section": "",
                    "route": ""
                })
                if st.session_state.get("offers"):
                    markets = ", ".join(st.session_state.offers.keys())
                    st.write(f"🛒 Used deals from: {markets}")
                status.update(label="✅ Done!", state="complete")
            response = result["messages"][-1].content

        except RateLimitError as e:
            wait_match = re.search(r"Please try again in (\d+h)?(\d+m)?([\d.]+s)?", str(e))
            if wait_match:
                wait_str = "".join(p for p in wait_match.groups() if p)
                response = f"⚠️ Rate limit reached. Please try again in **{wait_str}**."
            else:
                response = "⚠️ Rate limit reached. Please try again in a few minutes!"

        except Exception as e:
            response = f"⚠️ Something went wrong: {str(e)}"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})