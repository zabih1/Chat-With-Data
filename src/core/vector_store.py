from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
import os

from config import GEMINI_API_KEY, DOCUMENT_CHUNK_SIZE, DOCUMENT_CHUNK_OVERLAP

# In-memory storage for vector databases
VECTOR_STORES = {
    "document": None,
    "website": None
}

def create_embeddings():
    """Create and return the embedding model."""
    return GoogleGenerativeAIEmbeddings(
        google_api_key=GEMINI_API_KEY, 
        model="models/embedding-001"
    )

def split_content(documents):
    """Split documents into chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DOCUMENT_CHUNK_SIZE,
        chunk_overlap=DOCUMENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def setup_vector_database(docs, db_type="document"):
    """Set up and return an in-memory vector database with the given documents."""
    embeddings = create_embeddings()
    
    # Create in-memory vector database (no persist_directory)
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings
    )
    
    # Store the database in memory
    VECTOR_STORES[db_type] = vector_db
    
    return vector_db

def load_vector_database(db_type="document"):
    """Load an existing in-memory vector database."""
    vector_db = VECTOR_STORES.get(db_type)
    
    if vector_db is None:
        return None
    
    # Check if the database is empty
    try:
        if len(vector_db.get()['ids']) == 0:
            return None
    except Exception:
        return None
    
    return vector_db

def clear_vector_database(db_type="document"):
    """Clear an existing in-memory vector database."""
    vector_db = VECTOR_STORES.get(db_type)
    
    if vector_db is not None:
        try:
            all_ids = vector_db.get()['ids']
            if all_ids:
                vector_db._collection.delete(all_ids)
                print(f"Successfully deleted {len(all_ids)} documents from vector database.")
        except Exception as e:
            print(f"Error clearing vector database: {e}")
    
    # Reset the database to None
    VECTOR_STORES[db_type] = None
    
    return True