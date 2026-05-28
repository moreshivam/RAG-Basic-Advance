"""
RAG FUSION (Reciprocal Rank Fusion)
─────────────────────────────────────
Builds on Multi-Query by adding smarter ranking.

Multi-Query problem: just merges chunks naively — no sense of which
chunks are most relevant across all queries.

RAG Fusion solution: score every chunk using Reciprocal Rank Fusion (RRF).
Chunks that appear consistently at the top across multiple queries get
boosted. One-off appearances rank lower.

RRF formula: score(doc) = Σ 1 / (k + rank)
  - k = 60 (constant, prevents top-ranked docs dominating too much)
  - rank = position of doc in a single query's results (1 = best)
  - Σ = sum across all queries

Flow:
  question
    → LLM generates 3 alternative queries
    → retrieve chunks for each query (with their ranks)
    → RRF scores all chunks
    → top ranked chunks → LLM → final answer
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Step 1: Query generator (same as multi-query) ────────────────────────────
query_prompt = ChatPromptTemplate.from_template("""You are an AI assistant.
Generate 3 different versions of the given question to retrieve relevant documents.
Write one question per line. Do not number them. Do not add explanations.

Original question: {question}

3 alternative versions:""")

query_generator = (
    query_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().split("\n"))
)

# ── Step 2: Reciprocal Rank Fusion ────────────────────────────────────────────
def reciprocal_rank_fusion(queries: list[str], k: int = 60) -> list:
    """
    Retrieve docs for each query, score them with RRF, return sorted by score.

    k=60 is the standard constant from the original RRF paper.
    Higher k = smaller gap between ranks (more democratic scoring).
    """
    # scores maps: doc content → cumulative RRF score
    scores: dict[str, float] = {}
    # docs maps: doc content → actual document object (to return later)
    docs_map: dict[str, object] = {}

    for query in queries:
        if not query.strip():
            continue

        results = retriever.invoke(query.strip())

        # rank starts at 1 (best match), goes up
        for rank, doc in enumerate(results, start=1):
            content = doc.page_content

            if content not in docs_map:
                docs_map[content] = doc
                scores[content] = 0.0

            # RRF formula: add 1/(k + rank) to this doc's cumulative score
            scores[content] += 1.0 / (k + rank)

    # sort docs by score descending (highest score = most relevant)
    sorted_docs = sorted(docs_map.values(), key=lambda d: scores[d.page_content], reverse=True)
    return sorted_docs

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

# ── Step 4: Full RAG Fusion chain ─────────────────────────────────────────────
chain = (
    {
        "context": query_generator | reciprocal_rank_fusion | format_docs,
        "question": RunnablePassthrough()
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ── Step 5: Interactive Q&A loop ─────────────────────────────────────────────
if __name__ == "__main__":
    print("RAG Fusion ready!")
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

        print("\nRunning RRF ranking...")
        ranked_docs = reciprocal_rank_fusion(queries)
        print(f"  Ranked {len(ranked_docs)} unique chunks")
        print("\nTop chunks after RRF:")
        for i, doc in enumerate(ranked_docs[:3], 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"\n  [#{i}] {source} | Page {page}")
            print(f"  {doc.page_content[:200]}...")

        print("\n--- Answer ---")
        answer = chain.invoke(question)
        print(f"{answer}\n")
        print("-" * 60)
