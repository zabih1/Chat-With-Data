from dotenv import load_dotenv
import os

load_dotenv()

# 📌 API Keys 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
LLAMA_PARSER_API = os.getenv("LLAMA_PARSER_API")

# 📌 Database Config
DATABASE_URI = os.getenv("DATABASE_URI")

# 📌 Application Settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
PORT = int(os.getenv("PORT", 5000))
HOST = os.getenv("HOST", "0.0.0.0")


# 📌 Vector Database Settings
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
DOCUMENT_CHUNK_SIZE = 1000
DOCUMENT_CHUNK_OVERLAP = 200


# 📌 Model Settings
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
