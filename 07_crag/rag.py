"""
CRAG (Corrective Retrieval Augmented Generation)
──────────────────────────────────────────────────
Problem: basic RAG blindly trusts retrieved chunks even if they're irrelevant.
Result: LLM hallucinates when chunks don't contain the answer.

CRAG fix: grade every retrieved chunk → decide what to do:
  - RELEVANT chunks found   → use them as context
  - NO relevant chunks      → fall back to web search

Flow:
  question
    → retrieve chunks from vectorstore
    → grade each chunk (LLM: relevant or not?)
    → if relevant chunks exist  → use them
    → if no relevant chunks     → DuckDuckGo web search
    → feed context to LLM → final answer
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

# ── Step 1: Grader ────────────────────────────────────────────────────────────
# Reads a chunk + question → outputs "yes" (relevant) or "no" (not relevant)
grader_prompt = ChatPromptTemplate.from_template("""You are a grader checking if a
document is relevant to a question.

If the document contains information that helps answer the question, say "yes".
Otherwise say "no".

Answer with only "yes" or "no". Nothing else.

Document: {document}
Question: {question}

Relevant?""")

grader_chain = grader_prompt | llm | StrOutputParser()

def grade_documents(question: str, docs: list) -> list:
    """Filter docs — keep only those graded as relevant."""
    relevant = []
    for doc in docs:
        grade = grader_chain.invoke({
            "document": doc.page_content,
            "question": question
        })
        if "yes" in grade.strip().lower():
            relevant.append(doc)
    return relevant

# ── Step 2: Web search fallback ───────────────────────────────────────────────
def web_search(question: str) -> Document:
    """Search DuckDuckGo and return results as a single Document."""
    with DDGS() as ddgs:
        results = list(ddgs.text(question, max_results=3))

    web_content = "\n\n".join(
        f"{r['title']}\n{r['body']}" for r in results
    )
    return Document(
        page_content=web_content,
        metadata={"source": "web_search"}
    )

# ── Step 3: CRAG retriever — the core logic ───────────────────────────────────
def crag_retrieve(question: str) -> list[Document]:
    """
    1. Retrieve chunks from vectorstore
    2. Grade each chunk
    3. If relevant chunks exist → return them
    4. If none relevant → web search
    """
    print("\n  Retrieving from vectorstore...")
    docs = retriever.invoke(question)

    print("  Grading chunks...")
    relevant_docs = grade_documents(question, docs)

    if relevant_docs:
        print(f"  Grade: RELEVANT ✓ ({len(relevant_docs)}/{len(docs)} chunks passed)")
        return relevant_docs
    else:
        print(f"  Grade: IRRELEVANT ✗ — falling back to web search")
        web_doc = web_search(question)
        print(f"  Web search complete")
        return [web_doc]

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 4: Answer prompt ─────────────────────────────────────────────────────
answer_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:""")

# ── Step 5: Full CRAG chain ───────────────────────────────────────────────────
chain = (
    {
        "context": RunnableLambda(crag_retrieve) | format_docs,
        "question": RunnablePassthrough()
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ── Interactive Q&A loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CRAG ready!")
    print("Ask questions from your PDFs OR questions your PDFs don't cover.")
    print("Watch how it switches between vectorstore and web search.\n")
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
