from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

# ── Step 1: Load all PDFs from the data/ folder ──────────────────────────────
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()
print(f"  Loaded {len(documents)} page(s) from your PDFs")

# ── Step 2: Split into smaller chunks ────────────────────────────────────────
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)
print(f"  Created {len(chunks)} chunk(s)")

# ── Step 3: Embed chunks and store in ChromaDB (locally) ─────────────────────
# HuggingFace embeddings run locally — no API key needed
print("\nLoading embedding model (first run downloads ~80MB)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding and storing in vectorstore...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore",
)
print("  Done! Vectorstore saved to ./vectorstore/")
print("\nIndexing complete. You can now run rag.py to ask questions.")
