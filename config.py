from dotenv import load_dotenv
import os

load_dotenv()

# 📌 API Keys 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSER_API = os.getenv("LLAMA_PARSER_API")

# 📌 Database Config
DATABASE_URI = os.getenv("DATABASE_URI")

# 📌 Application Settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
PORT = int(os.getenv("PORT", 5000))
HOST = os.getenv("HOST", "0.0.0.0")

# proxy for the youtube video summary.
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")

# 📌 Vector Embedding Settings
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