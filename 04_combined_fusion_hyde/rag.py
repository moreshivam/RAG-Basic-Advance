"""
COMBINED: HyDE + RAG FUSION
────────────────────────────
Combines two techniques for maximum retrieval quality:

HyDE (Hypothetical Document Embeddings):
  Instead of embedding the question, ask LLM to write a hypothetical
  answer and embed THAT. Hypothetical answers are written like document
  chunks, so they match better in embedding space.

RAG Fusion (Reciprocal Rank Fusion):
  Generate multiple query versions, retrieve for each, then rank
  all chunks using RRF so the most consistently relevant ones bubble up.

Combined flow:
  question
    → generate 3 alternative queries          (LLM call #1)
    → for each query → generate hypothetical answer (LLM calls #2,#3,#4)
    → embed each hypothetical answer
    → retrieve 3 chunks per hypothetical      (3 vector searches)
    → RRF ranks all retrieved chunks
    → top chunks → LLM → final answer         (LLM call #5)

Cost: 5 LLM calls per question. Best retrieval quality.
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)

# ── Vectorstore ───────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="../vectorstore",
    embedding_function=embeddings,
)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Step 1: Query generator ───────────────────────────────────────────────────
query_prompt = ChatPromptTemplate.from_template("""You are an AI assistant.
Generate 3 different versions of the given question to retrieve relevant documents.
Write one question per line. Do not number them. Do not add explanations.

Original question: {question}

3 alternative versions:""")

query_generator = (
    query_prompt
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()])
)

# ── Step 2: Hypothetical answer generator (HyDE) ─────────────────────────────
hyde_prompt = ChatPromptTemplate.from_template("""Write a short paragraph that would
directly answer the following question, as if it came from a textbook or document.
Do not say "I think" or "based on". Just write the answer as a fact.

Question: {question}

Hypothetical answer:""")

hyde_generator = (
    hyde_prompt
    | llm
    | StrOutputParser()
)

# ── Step 3: Retrieve using hypothetical answer embeddings (HyDE search) ───────
def hyde_retrieve(query: str, k: int = 3) -> list:
    """Generate a hypothetical answer, embed it, search vectorstore with it."""
    hypothetical_answer = hyde_generator.invoke({"question": query})
    # embed the hypothetical answer and search vectorstore
    results = vectorstore.similarity_search(hypothetical_answer, k=k)
    return results

# ── Step 4: Reciprocal Rank Fusion ────────────────────────────────────────────
def reciprocal_rank_fusion(queries: list[str], k: int = 60) -> list:
    """
    For each query: generate hypothetical answer → retrieve chunks.
    Then RRF rank all chunks across all queries.
    """
    scores: dict[str, float] = {}
    docs_map: dict[str, object] = {}

    for query in queries:
        results = hyde_retrieve(query)

        for rank, doc in enumerate(results, start=1):
            content = doc.page_content

            if content not in docs_map:
                docs_map[content] = doc
                scores[content] = 0.0

            # RRF: 1/(k + rank) — chunks appearing high across many queries score higher
            scores[content] += 1.0 / (k + rank)

    sorted_docs = sorted(docs_map.values(), key=lambda d: scores[d.page_content], reverse=True)
    return sorted_docs

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 5: Answer prompt ─────────────────────────────────────────────────────
answer_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

# ── Step 6: Full chain ────────────────────────────────────────────────────────
chain = (
    {
        "context": query_generator | reciprocal_rank_fusion | format_docs,
        "question": RunnablePassthrough()
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ── Interactive Q&A loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("HyDE + RAG Fusion ready!")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\n[1/3] Generating alternative queries...")
        queries = query_generator.invoke(question)
        for i, q in enumerate(queries, 1):
            print(f"  Q{i}: {q}")

        print("\n[2/3] Generating hypothetical answers + retrieving chunks...")
        ranked_docs = reciprocal_rank_fusion(queries)
        print(f"  Ranked {len(ranked_docs)} unique chunks via HyDE + RRF")

        print("\nTop 3 chunks after ranking:")
        for i, doc in enumerate(ranked_docs[:3], 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"\n  [#{i}] {source} | Page {page}")
            print(f"  {doc.page_content[:200]}...")

        print("\n[3/3] Generating final answer...")
        answer = chain.invoke(question)
        print(f"\n--- Answer ---\n{answer}\n")
        print("-" * 60)
