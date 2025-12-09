from src.retriever.chroma_retriever import get_chroma_retriever

def query_retriever(retriever, query):
    """
    Safely query a VectorStoreRetriever, handling different LangChain versions.
    """
    try:
        # Try public method
        return retriever.get_relevant_documents(query)
    except AttributeError:
        # For newer versions requiring run_manager
        return retriever._get_relevant_documents(query, run_manager=None)
    except TypeError:
        # Some versions may throw TypeError if run_manager is missing
        return retriever._get_relevant_documents(query, run_manager=None)

def main():
    # Load retriever
    retriever = get_chroma_retriever(k=3)

    # Define your query
    query = "Eligibility criteria for MSc Data Science"

    # Get relevant documents
    results = query_retriever(retriever, query)

    # Print results
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:\n{doc.page_content}")

if __name__ == "__main__":
    main()
