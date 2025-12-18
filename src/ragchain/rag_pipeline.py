
from langchain_google_genai import ChatGoogleGenerativeAI



from src.ragchain.rag_utils import (
    get_reranker,
    retrieve_documents,
    format_context,
    TOP_K_FINAL,
    TEMPERATURE
)


def run_rag(query: str, retriever):
    # 1. Retrieve
    docs = retrieve_documents(retriever, query)

    # 2. Rerank
    reranker = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )
    ][:TOP_K_FINAL]

    # 3. Build context
    context = format_context(ranked_docs)

    # 4. Call LLM
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=TEMPERATURE
)


    prompt =f"""
You are an expert Admissions Counselor for Kerala University. Your goal is to provide accurate, helpful, and professional assistance to students.

Instructions:
1. **Analyze the Context**: Read the provided "Context Documents" carefully.
2. **Answer Based ONLY on Context**: Do not use outside knowledge. If the answer is not in the documents, say "I don't have that information in my current documents."
3. **Be Clear and Structured**: Use bullet points for steps or lists. Keep the tone welcoming but precise.
4. **Reasoning**: Briefly think about which document best answers the specific question.

Context Documents:
{context}

User Question: {query}

Answer:
"""
    return llm.invoke(prompt).content


  
from src.retriever.retriever import get_chroma_retriever

# Global retriever singleton
_RETRIEVER = None

def get_retriever_singleton():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = get_chroma_retriever(k=10)
    return _RETRIEVER

def query_rag(query: str):
    """
    Simplified entry point for RAG.
    Handles retriever initialization and runs the pipeline.
    """
    retriever = get_retriever_singleton()
    return run_rag(query, retriever)
