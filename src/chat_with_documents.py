import nest_asyncio
nest_asyncio.apply()

import os
import sys
from pathlib import Path
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import setup_vector_database
from app.prompt import chat_with_website_template
from app.llm import deepseek_r1_model, llama_model, gemini_model, mistral_model
from config import gemini_api_key



def create_rag_chain_with_prepared_data(session_data, model_name='llama'):

    prompt = ChatPromptTemplate.from_template(chat_with_website_template)
    
    vector_db = session_data.get('vector_db')
    
    if not vector_db and 'chunks' in session_data:
        vector_db = setup_vector_database(session_data['chunks'])
        session_data['vector_db'] = vector_db
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    if model_name == 'gemini':
        llm = gemini_model()
    elif model_name == 'mistral':
        llm = mistral_model()
    elif model_name == 'deepseek':
        llm = deepseek_r1_model()
    else:
        llm = llama_model()
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def load_vector_database():
    base_dir = Path(__file__).parent.parent
    persist_directory = os.path.join(base_dir, "data", "chroma_db")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=gemini_api_key, 
        model="models/embedding-001"
    )
    
    # Load the existing vector database
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_db


def chat_with_document(question, model_name='gemini'):
    try:
        vector_db = load_vector_database()
        
        prompt = ChatPromptTemplate.from_template(chat_with_website_template)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        if model_name == 'gemini':
            llm = gemini_model()
        elif model_name == 'mistral':
            llm = mistral_model()
        elif model_name == 'deepseek':
            llm = deepseek_r1_model()
        else:
            llm = llama_model()
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        raw_answer = rag_chain.invoke(question)
        
        return raw_answer
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise Exception(f"Error in chat_with_website: {str(e)}")
