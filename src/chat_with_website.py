import nest_asyncio
nest_asyncio.apply()

import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import SitemapLoader, WebBaseLoader, RecursiveUrlLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import setup_vector_database, split_content
from app.prompt import chat_with_website_template
from app.llm import deepseek_r1_model, llama_model, gemini_model, mistral_model
from config import gemini_api_key


def scrape_website_content(website_url):

    if website_url.endswith('/'):
        website_url = website_url[:-1]
        
    def html_extractor(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        return soup.get_text(separator=' ', strip=True)
    
    try:
        sitemap_url = f"{website_url}/sitemap.xml"
        sitemap_loader = SitemapLoader(
            web_path=sitemap_url,
            continue_on_failure=True,
            requests_per_second=2  
        )
        documents = sitemap_loader.load()

        if documents:
            print(f"Successfully loaded {len(documents)} pages from sitemap")
            return documents
    except Exception as e:
        print(f"Sitemap loading failed: {e}")
    
    try:
        print(f"Attempting recursive loading...")
        recursive_loader = RecursiveUrlLoader(
            url=website_url,
            extractor=html_extractor,
        )
        documents = recursive_loader.load()
        print(f"Successfully loaded {len(documents)} pages recursively")
        if documents:
            return documents
    except Exception as e:
        print(f"Recursive loading failed: {e}")
    
    try:
        print(f"Attempting to load main page as final fallback...")
        loader = WebBaseLoader(
            website_url,
            continue_on_failure=True,
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages with basic loader")
        return documents
    except Exception as e:
        print(f"All loading methods failed. Final error: {e}")
        return []


def prepare_website_data(website_url):

    try:
        documents = scrape_website_content(website_url)
        if not documents:
            return {'success': False, 'error': 'Failed to scrape website content'}
        
        chunks = split_content(documents)
        
        vector_db = setup_vector_database(chunks)
        
        return {
            'success': True,
            'message': 'Website content prepared successfully',
            'website_url': website_url,
            'vector_db': vector_db,
            'chunks': chunks
        }
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {'success': False, 'error': str(e)}


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
    
    # Initialize embeddings
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


def chat_with_website(website_url, question, model_name='deepseek'):
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
