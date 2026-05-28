"""
MULTI-QUERY RAG
───────────────
Problem with basic RAG: a single query might miss relevant chunks
if your question is phrased differently than the document.

Solution: Generate multiple versions of the same question,
retrieve chunks for each, merge all results, then answer.

Flow:
  question
    → LLM generates 3 alternative versions of the question
    → retrieve chunks for each version
    → merge all chunks (deduplicated)
    → feed to LLM → final answer
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)

# ── Vectorstore (same one built by 01_basic_rag/ingest.py) ───────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="../vectorstore",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Step 1: Query generator ───────────────────────────────────────────────────
# Takes the user's question and generates 3 alternative phrasings
query_prompt = ChatPromptTemplate.from_template("""You are an AI assistant.
Generate 3 different versions of the given question to retrieve relevant documents.
Write one question per line. Do not number them. Do not add explanations.

Original question: {question}

3 alternative versions:""")

query_generator = (
    query_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().split("\n"))  # split into list of questions
)

# ── Step 2: Multi-retriever ───────────────────────────────────────────────────
def retrieve_for_all_queries(queries: list[str]) -> list:
    """Run retrieval for each query and return deduplicated chunks."""
    seen = set()
    all_docs = []
    for query in queries:
        if not query.strip():
            continue
        docs = retriever.invoke(query.strip())
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    return all_docs

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 3: Answer prompt ─────────────────────────────────────────────────────
answer_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

# ── Step 4: Full multi-query chain ───────────────────────────────────────────
# Flow:
#   question → generate 3 queries → retrieve for each → merge → answer
chain = (
    {
        "context": query_generator | retrieve_for_all_queries | format_docs,
        "question": RunnablePassthrough()
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ── Step 5: Interactive Q&A loop ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Query RAG ready!")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\nGenerating alternative queries...")
        queries = query_generator.invoke(question)
        print("Generated queries:")
        for i, q in enumerate(queries, 1):
            if q.strip():
                print(f"  {i}. {q.strip()}")

        print("\nRetrieving and merging chunks...")
        all_docs = retrieve_for_all_queries(queries)
        print(f"  Found {len(all_docs)} unique chunks across all queries")

        print("\n--- Answer ---")
        answer = chain.invoke(question)
        print(f"{answer}\n")
        print("-" * 60)
