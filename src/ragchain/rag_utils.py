import os
from sentence_transformers import CrossEncoder

# ---------- CONFIG ----------
TOP_K_FINAL = 5
TEMPERATURE = 0.3

# ---------- RERANKER (cached) ----------
_RERANKER = None

def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


# ---------- SAFE RETRIEVAL ----------
def retrieve_documents(retriever, query):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


# ---------- FORMAT CONTEXT ----------
def format_context(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        text = doc.page_content.replace("\n", " ").strip()
        context += f"[Source {i}] {source}{page_info}\n{text}\n\n"
    return context
