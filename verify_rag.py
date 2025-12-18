import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

from src.ragchain.rag_pipeline import query_rag

def verify_rag_pipeline(query):
    print(f"\n❓ Question: {query}")
    print("-" * 50)
    
    try:
        print("🚀 Running RAG pipeline...")
        answer = query_rag(query)
        
        print("\n💡 Answer:")
        print(answer)
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "Chroma DB not found" in str(e):
            print("💡 Hint: Run the ingestion script first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RAG Pipeline")
    parser.add_argument("query", nargs="?", help="Question to ask", default="What documents are needed for admission?")
    
    args = parser.parse_args()
    verify_rag_pipeline(args.query)
