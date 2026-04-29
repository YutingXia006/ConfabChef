from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import load_retriever
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

load_dotenv(Path(__file__).parent.parent / ".env")

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
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        return str(response.content)

    return chain

if __name__ == "__main__":
    chain = build_chain()
    print(chain("Can you suggest a healthy Chinese dinner?"))