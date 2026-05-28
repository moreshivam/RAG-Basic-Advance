"""
MULTI-REPRESENTATION INDEXING — rag.py
────────────────────────────────────────
Retrieval flow:
  question
    → embed → search vectorstore (summaries) → find matching summaries
    → read doc_id from matched summary's metadata
    → fetch FULL doc from docstore.json using that doc_id
    → full doc goes to LLM as context
    → LLM generates answer
"""

import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv(override=True)

# ── Load vectorstore (summaries) ──────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="../vectorstore_summaries",
    embedding_function=embeddings,
)

# ── Load docstore (full documents) ────────────────────────────────────────────
with open("../docstore.json") as f:
    docstore = json.load(f)

# ── Custom retriever — searches summaries, returns full docs ──────────────────
def multi_vector_retrieve(question: str, k: int = 3) -> list[Document]:
    """
    1. Search vectorstore for matching summaries
    2. Read doc_id from each summary's metadata
    3. Fetch full doc from docstore using doc_id
    """
    # search summaries in vectorstore
    matched_summaries = vectorstore.similarity_search(question, k=k)

    full_docs = []
    for summary in matched_summaries:
        doc_id = summary.metadata.get("doc_id")
        if doc_id and doc_id in docstore:
            entry = docstore[doc_id]
            full_docs.append(
                Document(
                    page_content=entry["page_content"],
                    metadata=entry["metadata"]
                )
            )
    return full_docs

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

# ── RAG chain ─────────────────────────────────────────────────────────────────
from langchain_core.runnables import RunnableLambda

chain = (
    {
        "context": RunnableLambda(multi_vector_retrieve) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Interactive Q&A loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Representation RAG ready!")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\nSearching summaries → fetching full docs...\n")
        retrieved = multi_vector_retrieve(question)
        print(f"Retrieved {len(retrieved)} full document(s)")
        for i, doc in enumerate(retrieved, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"\n  [Doc {i}] {source} | Page {page}")
            print(f"  {doc.page_content[:300]}...")

        print("\n--- Answer ---")
        answer = chain.invoke(question)
        print(f"{answer}\n")
        print("-" * 60)
