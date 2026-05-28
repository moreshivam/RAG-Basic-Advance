from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)

# ── Step 1: Connect to the vectorstore we created in ingest.py ───────────────
# Must use the same embedding model as ingest.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
)

# Retriever: fetch top 3 most relevant chunks for any question
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Step 2: Prompt template ───────────────────────────────────────────────────
template = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# ── Step 3: LLM (Gemini Flash - free tier) ───────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Step 4: Build the RAG chain ───────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Step 5: Interactive Q&A loop ──────────────────────────────────────────────
if __name__ == "__main__":
    print("RAG system ready! Ask questions about your PDFs.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\nSearching your documents...\n")

        # Show which chunks were retrieved
        retrieved_docs = retriever.invoke(question)
        print("--- Retrieved chunks from your PDFs ---")
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"\n[Chunk {i}] Source: {source} | Page: {page}")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print("\n--- Answer ---")

        answer = chain.invoke(question)
        print(f"{answer}\n")
        print("-" * 60)
