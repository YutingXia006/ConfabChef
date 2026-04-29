from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from src.chat import build_chain
from src.offers import load_or_fetch_offers

load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(
    page_title="ConfabChef",
    page_icon="🍳",
    layout="centered"
)

st.title("🍳 ConfabChef (CC)")
st.caption("Your AI sous-chef — because even AI chefs occasionally confabulate")

FAISS_INDEX = Path("data/faiss_index")

if not FAISS_INDEX.exists():
    with st.status("📚 Building recipe database for the first time...", expanded=True) as status:
        st.write("Loading recipes...")
        from src.ingest import load_csv_recipes, load_txt_recipes, split_documents, build_faiss_index
        csv_docs = load_csv_recipes()
        txt_docs = load_txt_recipes()
        all_docs = csv_docs + txt_docs
        st.write(f"Found {len(all_docs)} recipes!")
        st.write("Building search index (this may take a few minutes)...")
        chunks = split_documents(all_docs)
        build_faiss_index(chunks)
        status.update(label="✅ Recipe database ready!", state="complete")

if "chain" not in st.session_state:
    with st.status("🍳 Preparing the kitchen...", expanded=True) as status:
        st.write("Loading recipe database...")
        st.write("Initializing AI model...")
        st.session_state.chain = build_chain()
        status.update(label="✅ ConfabChef is ready!", state="complete")

if st.button("🛒 Use this week's supermarket deals"):
    with st.spinner("Loading deals..."):
        offers, message = load_or_fetch_offers()
        st.session_state.offers = offers
        st.success(message)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me for a recipe or meal plan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Cooking up an answer..."):
            response = st.session_state.chain(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})