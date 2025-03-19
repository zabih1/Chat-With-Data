from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import weaviate
from weaviate.classes.init import Auth
from config import api_key,weaviate_url, weaviate_api_key


def setup_vector_database(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key, 
        model="models/embedding-001"
    )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        skip_init_checks=True
    )

    vector_db = WeaviateVectorStore.from_documents(docs, embeddings, client=client)
    return vector_db