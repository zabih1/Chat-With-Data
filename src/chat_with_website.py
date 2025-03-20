import nest_asyncio
nest_asyncio.apply()

import os, sys
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.sitemap import SitemapLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import split_content, setup_vector_database
from app.prompt import chat_with_website_template
from app.llm import deepseek_r1_model


def load_website_content(website_url):
    
    if website_url.endswith('/'):
        website_url = website_url[:-1]
    
    sitemap_url = f"{website_url}/sitemap.xml"
    
    sitemap_loader = SitemapLoader(web_path=sitemap_url)
    
    documents = sitemap_loader.load()
    
    print(f"Successfully loaded {len(documents)} pages from {website_url}")
    return documents



def create_rag_chain(docs):
    
    prompt = ChatPromptTemplate.from_template(chat_with_website_template)

    vector_db = setup_vector_database(docs)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = deepseek_r1_model()
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


website_content = load_website_content("https://www.devsinc.com/")

docs = split_content(website_content)

chain = create_rag_chain(docs)

result = chain.invoke("who is asif usman?")

print(result)