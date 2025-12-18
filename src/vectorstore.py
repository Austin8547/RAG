import os
from langchain_community.vectorstores import Chroma

_VECTOR_DB = None  # global

def create_chroma_db(
    chunks,
    embedding_model,
    persist_directory="chroma_db",
    overwrite=True
):
    global _VECTOR_DB

    if os.path.exists(persist_directory) and not overwrite:
        print("Using existing Chroma DB at", persist_directory)
        _VECTOR_DB = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        return _VECTOR_DB

    _VECTOR_DB = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("ChromaDB created and embeddings stored successfully.")
    return _VECTOR_DB

