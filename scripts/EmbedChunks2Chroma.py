from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pathlib import Path
import logging
import os

MARKDOWN_DIR = Path("Outputs/markdown")
VECTOR_DB_DIR = Path("Outputs/vector_db")

def load_markdown_files():
    docs = []
    for file in MARKDOWN_DIR.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": str(file)}))
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)

def embed_and_store_chunks(chunks):
    # Use the updated HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Faster, smaller model
    )
    
    # Convert Path to string for ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)  # Convert to string
    )
    vector_db.persist()
    print(f"Stored {len(chunks)} chunks in the vector database at {VECTOR_DB_DIR}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting to embed and store chunks from markdown files")
    
    # Create output directory if it doesn't exist
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    docs = load_markdown_files()
    logger.info(f"Loaded {len(docs)} documents from markdown files")
    chunks = chunk_documents(docs)
    logger.info(f"Chunked documents into {len(chunks)} chunks")
    embed_and_store_chunks(chunks)
    logger.info("Embedding and storing chunks completed")
