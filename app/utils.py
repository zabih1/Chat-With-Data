from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model

import weaviate
from weaviate.classes.init import Auth
from config import gemini_api_key, groq_api_key ,weaviate_url, weaviate_api_key

def gemini_model():
    llm = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.0-flash"
)

def llama_model():
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    return llm

llm = llama_model()
llm.invoke("tell me a joke")

def mistral_model():
    llm = init_chat_model("mistral", model_provider="groq")

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