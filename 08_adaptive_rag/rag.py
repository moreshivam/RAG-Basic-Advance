"""
ADAPTIVE RAG + CRAG (Combined)
───────────────────────────────
Adaptive RAG decides strategy BEFORE retrieval.
CRAG grades chunks AFTER retrieval and corrects if needed.

Full flow:
  question
    → router classifies: simple / complex / web_search
    ↓
    if web_search  → go straight to web (skip vectorstore)
    if simple      → basic retrieval   → grade chunks
    if complex     → multi-query       → grade chunks
                                             ↓
                              relevant? → use them
                              irrelevant? → web search fallback
    ↓
  LLM generates final answer
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from ddgs import DDGS

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

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE RAG — router
# ─────────────────────────────────────────────────────────────────────────────
router_prompt = ChatPromptTemplate.from_template("""You are a question classifier.
Classify the question into exactly one category. Reply with only the category name.

Categories:
- simple      (straightforward factual question, one clear answer)
- complex     (requires comparing, multi-step reasoning, or broad coverage)
- web_search  (current events, recent news, or unrelated to AI/ML topics)

Examples:
  "What is RAG?"                          → simple
  "Compare RAG vs fine-tuning in detail"  → complex
  "Who won IPL 2024?"                     → web_search

Question: {question}
Category:""")

router_chain = (
    router_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().lower())
)

# ─────────────────────────────────────────────────────────────────────────────
# CRAG — grader + web fallback
# ─────────────────────────────────────────────────────────────────────────────
grader_prompt = ChatPromptTemplate.from_template("""You are a grader checking if a
document is relevant to a question.
Answer with only "yes" or "no". Nothing else.

Document: {document}
Question: {question}
Relevant?""")

grader_chain = grader_prompt | llm | StrOutputParser()

def grade_documents(question: str, docs: list) -> list:
    """Keep only chunks graded as relevant."""
    relevant = []
    for doc in docs:
        grade = grader_chain.invoke({
            "document": doc.page_content,
            "question": question
        })
        if "yes" in grade.strip().lower():
            relevant.append(doc)
    return relevant

def web_search(question: str) -> list[Document]:
    """DuckDuckGo fallback when no relevant chunks found."""
    with DDGS() as ddgs:
        results = list(ddgs.text(question, max_results=3))
    web_content = "\n\n".join(f"{r['title']}\n{r['body']}" for r in results)
    return [Document(page_content=web_content, metadata={"source": "web_search"})]

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval strategies
# ─────────────────────────────────────────────────────────────────────────────
def simple_retrieve(question: str) -> list[Document]:
    """Single query retrieval."""
    return retriever.invoke(question)

query_gen_prompt = ChatPromptTemplate.from_template("""Generate 3 different versions
of the question. One per line. No numbering.

Question: {question}
3 versions:""")

query_generator = (
    query_gen_prompt
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()])
)

def complex_retrieve(question: str) -> list[Document]:
    """Multi-query retrieval with deduplication."""
    queries = query_generator.invoke({"question": question})
    print(f"  Generated queries:")
    for q in queries:
        print(f"    - {q}")

    seen = set()
    all_docs = []
    for query in queries:
        for doc in retriever.invoke(query):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    return all_docs

# ─────────────────────────────────────────────────────────────────────────────
# Combined pipeline — Adaptive RAG + CRAG
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_crag_retrieve(question: str) -> list[Document]:
    """
    Step 1 (Adaptive): classify question → pick retrieval strategy
    Step 2 (CRAG):     grade retrieved chunks → fallback to web if needed
    """
    # ── Adaptive: classify ────────────────────────────────────────────────────
    strategy = router_chain.invoke({"question": question})
    if strategy not in ("simple", "complex", "web_search"):
        strategy = "simple"

    print(f"  [Adaptive] Strategy: {strategy.upper()}")

    # ── web_search: skip vectorstore entirely ─────────────────────────────────
    if strategy == "web_search":
        print("  [Adaptive] Going straight to web search")
        return web_search(question)

    # ── simple or complex: retrieve then grade (CRAG) ─────────────────────────
    if strategy == "simple":
        docs = simple_retrieve(question)
    else:
        docs = complex_retrieve(question)

    # ── CRAG: grade the retrieved chunks ──────────────────────────────────────
    print(f"  [CRAG] Grading {len(docs)} chunks...")
    relevant_docs = grade_documents(question, docs)

    if relevant_docs:
        print(f"  [CRAG] RELEVANT ✓ ({len(relevant_docs)}/{len(docs)} passed)")
        return relevant_docs
    else:
        print(f"  [CRAG] IRRELEVANT ✗ — falling back to web search")
        return web_search(question)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Answer prompt ─────────────────────────────────────────────────────────────
answer_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:""")

# ── Full chain ────────────────────────────────────────────────────────────────
chain = (
    {
        "context": RunnableLambda(adaptive_crag_retrieve) | format_docs,
        "question": RunnablePassthrough()
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ── Interactive Q&A loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Adaptive RAG + CRAG ready!")
    print("Try all three question types:\n")
    print("  Simple:     'What is RAG?'")
    print("  Complex:    'Compare RAG and fine-tuning in detail'")
    print("  Web search: 'Who won IPL 2024?'\n")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print()
        answer = chain.invoke(question)
        print(f"\n--- Answer ---\n{answer}\n")
        print("-" * 60)
