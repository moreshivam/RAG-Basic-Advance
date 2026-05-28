"""
MULTI-REPRESENTATION INDEXING — ingest.py
──────────────────────────────────────────
Builds TWO stores:

1. VECTORSTORE — stores embeddings of SHORT SUMMARIES (for precise searching)
2. DOCSTORE    — stores FULL document pages as a JSON file (for rich LLM context)

Each summary is linked to its full doc via a shared ID.
Run this once before running rag.py.
"""

import uuid
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv(override=True)

# ── Load PDFs ─────────────────────────────────────────────────────────────────
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("../data/")
docs = loader.load()
print(f"  Loaded {len(docs)} pages")

# ── LLM for generating summaries ──────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

summary_prompt = ChatPromptTemplate.from_template("""Summarize the following document
in 1-2 concise sentences. Focus on the key topic and main point only.

Document:
{doc}

Summary:""")

summary_chain = summary_prompt | llm | StrOutputParser()

# ── Generate a unique ID for each page ───────────────────────────────────────
doc_ids = [str(uuid.uuid4()) for _ in docs]

# ── Generate summaries ────────────────────────────────────────────────────────
print(f"\nGenerating summaries for {len(docs)} pages...")
summary_docs = []
for i, doc in enumerate(docs):
    print(f"  Summarizing page {i+1}/{len(docs)}...", end="\r")
    summary = summary_chain.invoke({"doc": doc.page_content})
    summary_docs.append(
        Document(
            page_content=summary,
            metadata={"doc_id": doc_ids[i]}
        )
    )
print(f"\n  Generated {len(summary_docs)} summaries")

# ── Store summaries in vectorstore ────────────────────────────────────────────
print("\nStoring summaries in vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=summary_docs,
    embedding=embeddings,
    persist_directory="../vectorstore_summaries",
)
print("  Saved to ../vectorstore_summaries/")

# ── Store full docs in a JSON file (docstore) ─────────────────────────────────
print("\nSaving full documents to docstore...")
docstore = {
    doc_ids[i]: {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }
    for i, doc in enumerate(docs)
}
with open("../docstore.json", "w") as f:
    json.dump(docstore, f)
print("  Saved to ../docstore.json")

print("\nIndexing complete! Run rag.py to ask questions.")
