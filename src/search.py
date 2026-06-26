import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "qwen/qwen3-32b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Check if vectorstore indices already exist, otherwise build
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
            
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model_name=llm_model)
        print(f"[INFO] GROQ LLM Initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        prompt = f"""Summarize the following context for the query: '{query}'

Context:
{context}

Summary:"""
        response = self.llm.invoke([prompt])
        return response.content
    

# Example Usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is the Python Language?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)