import nest_asyncio
nest_asyncio.apply()

import os, sys
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import SitemapLoader, WebBaseLoader, RecursiveUrlLoader , TextLoader
from bs4 import BeautifulSoup
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import setup_vector_database
from app.prompt import chat_with_website_template
from app.llm import deepseek_r1_model ,llama_model, gemini_model, mistral_model


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

def save_website_content(website_content, output_dir='data'):

    os.makedirs(output_dir, exist_ok=True)
    
    text = []
    for i in website_content:
        clean_content = ' '.join(i.page_content.split())
        text.append(clean_content)
    
    if website_content and hasattr(website_content[0], 'metadata') and 'source' in website_content[0].metadata:
        url = website_content[0].metadata['source']
        filename = url.replace('://', '_').replace('/', '_')
        filename = filename[:150] + '.txt'  # Limit length
    else:
        filename = f"website_content_{int(time.time())}.txt"
    
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n\n'.join(text))

    return file_path

def load_website_content(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Website content file not found: {file_path}")
     
    loader = TextLoader(file_path)
    documents = loader.load()

    return documents

def create_rag_chain(docs, model_name='llama'):

    prompt = ChatPromptTemplate.from_template(chat_with_website_template)

    vector_db = setup_vector_database(docs)

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