import sys
import os

# Fix import path so "src" works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.retriever.chroma_retriever import get_chroma_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL


def safe_retrieve(retriever, query):
    """
    Works for all LangChain versions:
    - New versions use get_relevant_documents()
    - Older versions use _get_relevant_documents()
    """
    try:
        return retriever.get_relevant_documents(query)
    except Exception:
        return retriever._get_relevant_documents(query, run_manager=None)


def run_chain(query: str):
    # Load retriever
    retriever = get_chroma_retriever()

    # Retrieve documents safely
    docs = safe_retrieve(retriever, query)

    # Combine all context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Load LLM from config
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    prompt = f"""
You are an expert RAG assistant. Use ONLY the given context to answer.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content  # <-- FIXED: return only text

