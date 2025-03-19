
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL")

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.0-flash"
)


