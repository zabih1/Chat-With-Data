import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import gemini_api_key



def gemini_model():
    llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-2.0-flash")
    return llm  

def llama_model():
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    return llm

def mistral_model():
    llm = init_chat_model("mixtral-8x7b-32768", model_provider="groq")
    return llm

def deepseek_r1_model():
    llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
    return llm

