import nest_asyncio
nest_asyncio.apply()

import os
import sys
import shutil
from pathlib import Path
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from llama_cloud_services import LlamaParse
from groq import Groq


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from core modules
from src.core.prompts import DOCUMENT_CHAT_TEMPLATE
from src.core.llm import get_llm
from src.core.vector_store import split_content, setup_vector_database, load_vector_database
from config import LLAMA_PARSER_API

def load_document(file_path):
    """Load and process a document file."""
    try:
        parsed_documents = LlamaParse(
            api_key=LLAMA_PARSER_API, 
            premium_mode=True,
            result_type="markdown"
        ).load_data(file_path)
        
        # Convert to LangChain document format
        documents = [Document(page_content=doc.text) for doc in parsed_documents]
        
        # Split into chunks and create vector database
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

def chat_with_document(question, model_name='deepseek'):
    """Process a question against loaded documents."""
    try:
        vector_db = load_vector_database()
        
        if not vector_db:
            return "No documents have been uploaded yet. Please upload a document first."
        
        prompt = ChatPromptTemplate.from_template(DOCUMENT_CHAT_TEMPLATE)
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        llm = get_llm(model_name)
        
        # Create and execute the RAG chain
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
    vector_db = load_vector_database(db_type="document")
    
    if vector_db and hasattr(vector_db, '_collection'):
        all_ids = vector_db.get()['ids']
        
        if all_ids:
            vector_db._collection.delete(all_ids)
            print(f"Successfully deleted {len(all_ids)} documents from vector database.")
    
    # Add document removal from uploads folder
    try:
        base_dir = Path(__file__).parent.parent.parent
        uploads_dir = os.path.join(base_dir, "uploads")
        
        if os.path.exists(uploads_dir):
            for file in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
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
    



client = Groq()

def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file
        )

        return transcript.text
    
    