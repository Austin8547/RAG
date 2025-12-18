import os
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_PATH
from src.embeddings import get_embedding_model

embeddings = get_embedding_model()

def get_chroma_retriever(k=7):
    """
    Load Chroma vectorstore and return a retriever.

    Args:
        k (int): Number of top results to fetch for each query

    Returns:
        VectorStoreRetriever: Retriever object
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Chroma DB not found at {CHROMA_PATH}. Run ingestion first!")

    # Load Chroma vectorstore
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever