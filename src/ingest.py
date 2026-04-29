from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Paths
ROOT = Path(__file__).parent.parent
RECIPES_DIR = ROOT / "data" / "recipes"
FAISS_INDEX = ROOT / "data" / "faiss_index"
CSV_PATH = ROOT / "data" / "recipes" / "recipes_db.csv"

def load_csv_recipes():
    df = pd.read_csv(CSV_PATH, encoding='utf-8', encoding_errors='replace')
    documents = []
    for _, row in df.iterrows():
        content = f"""Name: {row['recipe_name']}
        Cuisine: {row['cuisine']}
        Ingredients: {row['ingredients']}
        Cooking Time: {row.get('cooking_time_minutes', 'N/A')} minutes
        Prep Time: {row.get('prep_time_minutes', 'N/A')} minutes
        Servings: {row.get('servings', 'N/A')}
        Calories per Serving: {row.get('calories_per_serving', 'N/A')}
        Dietary Restrictions: {row.get('dietary_restrictions', 'N/A')}"""
        
        documents.append(Document(
            page_content=content,
            metadata={"source": "kaggle_dataset", "cuisine": row['cuisine']}
        ))
    print(f"Loaded {len(documents)} recipes from CSV")
    return documents

def load_txt_recipes():
    loader = DirectoryLoader(
        str(RECIPES_DIR),
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} recipes")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks):
    print("Building FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="ibm-granite/granite-embedding-278m-multilingual"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX))
    print(f"FAISS index saved to {FAISS_INDEX}")

if __name__ == "__main__":
    csv_docs = load_csv_recipes()
    txt_docs = load_txt_recipes()
    all_docs = csv_docs + txt_docs
    chunks = split_documents(all_docs)
    build_faiss_index(chunks)