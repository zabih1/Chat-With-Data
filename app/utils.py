import sys
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import gemini_api_key


MODELS = [
    {"id": "llama", "name": "Llama 3"},
    {"id": "gemini", "name": "Gemini"},
    {"id": "mistral", "name": "Mistral"},
    {"id": "deepseek", "name": "DeepSeek"}
]

def get_model_display_name(model_id):
    for model in MODELS:
        if model["id"] == model_id:
            return model["name"]
    return model_id  


# >>>>>>>>>> 📌 Split documents. <<<<<<<<<<<<<<<<<<<<<<

def split_content(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    print("chunks is created")

    return chunks


# >>>>>>>>>> 📌 Setup vector database. <<<<<<<<<<<<<<<<<<<<<<
def setup_vector_database(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=gemini_api_key, 
        model="models/embedding-001"
    )
    base_dir = Path(__file__).parent.parent

    persist_directory = os.path.join(base_dir, "data", "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)

    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )

    print(f"Vector database set up at: {persist_directory}")
    return vector_db




