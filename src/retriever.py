from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

ROOT = Path(__file__).parent.parent
FAISS_INDEX = ROOT / "data" / "faiss_index"

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="ibm-granite/granite-embedding-278m-multilingual",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    retriever = load_retriever()
    results = retriever.invoke("I want something spicy with chicken")
    for doc in results:
        print("---")
        print(doc.page_content)