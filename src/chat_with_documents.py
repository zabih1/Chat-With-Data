import nest_asyncio
nest_asyncio.apply()

import os
import sys
import shutil
from pathlib import Path
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import setup_vector_database, split_content
from app.prompt import chat_with_document_template
from app.llm import deepseek_r1_model, llama_model, gemini_model, mistral_model
from config import gemini_api_key


def load_document(file_path):

    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            loader = PyMuPDFLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path)
        elif file_extension == 'docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            return {
                'success': False,
                'error': f'Unsupported file type: {file_extension}'
            }
        
        documents = loader.load()
        
        if not documents:
            return {
                'success': False,
                'error': 'No content could be extracted from the document'
            }
        
        chunks = split_content(documents)
        
        vector_db = setup_vector_database(chunks)
        
        return {
            'success': True,
            'message': 'Document loaded and processed successfully',
            'file_path': file_path,
            'chunks': chunks,
            'vector_db': vector_db
        }
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }


def create_rag_chain_with_documents(model_name='llama'):
    prompt = ChatPromptTemplate.from_template(chat_with_document_template)
    
    vector_db = load_vector_database()
    
    if not vector_db:
        raise ValueError("Vector database could not be loaded")
    
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
    
    # Check if the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=gemini_api_key, 
        model="models/embedding-001"
    )
    
    try:
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        if len(vector_db.get()['ids']) == 0:
            return None
        
        return vector_db
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None


def chat_with_document(question, model_name='deepseek'):

    try:
        vector_db = load_vector_database()
        
        if not vector_db:
            return "No documents have been uploaded yet. Please upload a document first."
        
        prompt = ChatPromptTemplate.from_template(chat_with_document_template)
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
        raise Exception(f"Error in chat_with_document: {str(e)}")


def clear_documents():

    try:
        base_dir = Path(__file__).parent.parent
        persist_directory = os.path.join(base_dir, "data", "chroma_db")
        
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        
        os.makedirs(persist_directory, exist_ok=True)
        
        # Also clear the uploads directory
        uploads_dir = os.path.join(base_dir, "uploads")
        if os.path.exists(uploads_dir):
            for file in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        return {
            'success': True,
            'message': 'All documents have been cleared'
        }
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }