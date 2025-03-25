from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
import os

from config import GEMINI_API_KEY, DOCUMENT_CHUNK_SIZE, DOCUMENT_CHUNK_OVERLAP

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
    """Set up and return a vector database with the given documents."""
    embeddings = create_embeddings()
    base_dir = Path(__file__).parent.parent.parent
    
    # Create directory based on the type of vector database
    if db_type == "website":
        persist_directory = os.path.join(base_dir, "./data/chroma_db_website")
    else:
        persist_directory = os.path.join(base_dir, "./data/chroma_db_document")
    
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create and return the vector database
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    return vector_db

def load_vector_database(db_type="document"):
    """Load an existing vector database."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Select the directory based on the type of vector database
    if db_type == "website":
        persist_directory = os.path.join(base_dir, "./data/chroma_db_website")
    else:
        persist_directory = os.path.join(base_dir, "./data/chroma_db_document")
    
    # Initialize embeddings
    embeddings = create_embeddings()
    
    try:
        # Load the existing vector database
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Check if the database is empty
        if len(vector_db.get()['ids']) == 0:
            return None
        
        return vector_db
        
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None