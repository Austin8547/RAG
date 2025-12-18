from dotenv import load_dotenv
load_dotenv()   # MUST be first

import os

from src.config import CHROMA_PATH
from src.ingestion import load_all_documents
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_chroma_db, get_chroma_retriever
from src.ragchain.rag_pipeline import run_rag

def main():
    # 1. Load documents
    docs = load_all_documents()
    print("Total documents loaded:", len(docs))

    # 2. Chunk documents
    chunks = chunk_documents(docs)
    print("Total chunks created:", len(chunks))

    # 3. Load embedding model
    embedding_model = get_embedding_model()
    print("Embedding model loaded.")

    # 4. Create or reuse ChromaDB (MUST be first)
    create_chroma_db(
        chunks,
        embedding_model,
        persist_directory=CHROMA_PATH,
        overwrite=True
    )
    print(f"ChromaDB ready at {CHROMA_PATH}")

    # 5. Create retriever (AFTER DB exists)
    retriever = get_chroma_retriever(k=10)
    print("Retriever created.")

if __name__ == "__main__":
    main()
