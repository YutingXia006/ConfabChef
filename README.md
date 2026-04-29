# 🍳 ConfabChef (CC)

> Your AI sous-chef — because even AI chefs occasionally confabulate.

ConfabChef is a RAG-based (Retrieval-Augmented Generation) recipe and meal planning chatbot built with LangChain, Groq, and Streamlit. It retrieves relevant recipes from a local knowledge base and generates personalised meal suggestions using a large language model.

## Demo

![ConfabChef Demo](docs/demo.png)

## How It Works

1. **Ingest** — Recipe documents are loaded, split into chunks and embedded into a FAISS vector index
2. **Retrieve** — User questions are matched against the vector index to find relevant recipes
3. **Generate** — A Groq LLM generates a personalised response based on the retrieved recipes

## Tech Stack

| Component | Technology |
| --- | --- |
| LLM | Groq (LLaMA 3.3 70B) |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace sentence-transformers (ibm-granite/granite-embedding-278m-multilingual) |
| UI | Streamlit |
| Language | Python 3.12 |

## Project Structure

``` text
ConfabChef/
├── src/
│   ├── ingest.py       # Load recipes & build FAISS index
│   ├── retriever.py    # Load FAISS index & retrieve relevant recipes
│   └── chat.py         # LLM chain with RAG prompt
├── data/
│   └── recipes/        # Recipe knowledge base (.txt files and .csv database)
├── app.py              # Streamlit UI
├── .env                # API keys (not committed)
└── requirements.txt
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YutingXia006/ConfabChef.git
cd ConfabChef
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up `.env`

GROQ_API_KEY=your_key_here

### 4. Build the FAISS index

```bash
python src/ingest.py
```

### 5. Run the app

```bash
streamlit run app.py
```

## Adding Your Own Recipes

Simply add `.txt` files to `data/recipes/` in this format:\
Name: Your Recipe Name\
Cuisine: Chinese\
Servings: 2\
Calories: 300\
Ingredients:

ingredient 1\
ingredient 2

Instructions:

Step one\
Step two

Tags: tag1, tag2

Then rebuild the index:

```bash
python src/ingest.py
```

## Author

Yuting Xia — [GitHub](https://github.com/YutingXia006) | [LinkedIn](https://www.linkedin.com/in/yuting-xia-89180a274/)

## Licence

MIT License — do whatever you want with it.

## Dataset

Recipe data sourced from [Collection of Recipes around the world](https://www.kaggle.com/datasets/prajwaldongre/collection-of-recipes-around-the-world) by Prajwal Dongre on Kaggle, licensed under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
