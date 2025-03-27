from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model


from config import GEMINI_API_KEY, GROQ_API_KEY


def get_llm(model_name="deepseek"):
    """Initialize and return the specified language model."""
    if model_name == "gemini":
        return ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-2.0-flash") 
    elif model_name == "llama":
        return init_chat_model("llama3-8b-8192", model_provider="groq")
    
    elif model_name == "mistral":
        return init_chat_model("mistral-saba-24b", model_provider="groq")
    
    elif model_name == "deepseek":
        return init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
    
    else:
        return init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
    
