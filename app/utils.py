import weaviate
from weaviate.classes.init import Auth
import sys
import os
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import gemini_api_key, weaviate_url, weaviate_api_key



MODELS = [
    {"id": "llama", "name": "Llama 3", "description": "A fast and efficient model for general summaries"},
    {"id": "gemini", "name": "Gemini", "description": "Google's Gemini model for detailed analysis"},
    {"id": "mistral", "name": "Mistral", "description": "Specialized for concise and accurate summaries"},
    {"id": "deepseek", "name": "DeepSeek", "description": "Optimized for deep technical content"}
]

MODEL_DISPLAY_NAMES = {
    'llama': 'Llama 3',
    'gemini': 'Gemini',
    'mistral': 'Mistral',
    'deepseek': 'DeepSeek'
}

def process_markdown(text):
    processed = text.replace('• ', '* ')
    processed = processed.replace('**Title:**', '\n**Title:**')
    processed = processed.replace('**Main Topic:**', '\n**Main Topic:**')
    processed = processed.replace('**Key Points:**', '\n**Key Points:**')
    processed = processed.replace('**Conclusion:**', '\n**Conclusion:**')
    return processed
# >>>>>>>>>> 📌 Split documents. <<<<<<<<<<<<<<<<<<<<<<

def split_content(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    return chunks


# >>>>>>>>>> 📌 Setup vector database. <<<<<<<<<<<<<<<<<<<<<<
def setup_vector_database(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=gemini_api_key, 
        model="models/embedding-001"
    )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        skip_init_checks=True
    )

    vector_db = WeaviateVectorStore.from_documents(docs, embeddings, client=client)
    return vector_db



def process_markdown(text):
    """Process special markdown elements for proper rendering"""
    processed = text.replace('• ', '* ')
    
    # Add line breaks before headings if they don't exist
    processed = processed.replace('**Title:**', '\n**Title:**')
    processed = processed.replace('**Main Topic:**', '\n**Main Topic:**')
    processed = processed.replace('**Key Points:**', '\n**Key Points:**')
    processed = processed.replace('**Conclusion:**', '\n**Conclusion:**')
    
    return processed
