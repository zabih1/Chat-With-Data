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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.vector_store import split_content, setup_vector_database, load_vector_database

from src.core.prompts import WEBSITE_CHAT_TEMPLATE
from src.core.llm import get_llm


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
        
        vector_db = setup_vector_database(chunks, db_type="website")
        
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

    prompt = ChatPromptTemplate.from_template(WEBSITE_CHAT_TEMPLATE)
    
    vector_db = session_data.get('vector_db')
    
    if not vector_db and 'chunks' in session_data:
        vector_db = setup_vector_database(session_data['chunks'], db_type="website")
        session_data['vector_db'] = vector_db
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    llm = get_llm(model_name)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def chat_with_website(website_url, question, model_name='deepseek'):
    try:
        vector_db = load_vector_database(db_type="website")
        
        prompt = ChatPromptTemplate.from_template(WEBSITE_CHAT_TEMPLATE)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        llm = get_llm(model_name)

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
