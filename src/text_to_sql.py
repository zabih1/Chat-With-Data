from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
import markdown
import re
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.prompt import text_to_sql_templates
from app.llm import gemini_model, mistral_model, deepseek_r1_model, llama_model

# Initialize models
llm1 = gemini_model()
llm2 = mistral_model()
llm3 = deepseek_r1_model()
llm4 = llama_model()

query_prompt_template = PromptTemplate.from_template(text_to_sql_templates)

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    db_uri: str

def write_query(state: State) -> dict:
    try:
        # Check if db_uri exists in state and is not None
        if "db_uri" not in state or not state["db_uri"]:
            return {"query": "-- Error: Database URI is missing or empty"}
        
        db = SQLDatabase.from_uri(state["db_uri"])
        
        if not db:
            return {"query": "-- Error: Database not available"}
        
        prompt = query_prompt_template.invoke({
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        })

        response = llm1.invoke(prompt)
        
        # Extract SQL query from response
        sql_query = response.content if hasattr(response, "content") else str(response)
        
        if "```sql" in sql_query:
            sql_match = re.search(r"```sql\n(.*?)\n```", sql_query, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1)
        elif "```" in sql_query:
            sql_match = re.search(r"```\n(.*?)\n```", sql_query, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1)
        
        return {"query": sql_query}
    
    except Exception as e:
        return {"query": f"-- Error generating SQL query: {str(e)}"}

def execute_query(state: State) -> dict:
    try:
        # Check if db_uri exists in state and is not None
        if "db_uri" not in state or not state["db_uri"]:
            return {"result": "Error: Database URI is missing or empty"}
        
        db = SQLDatabase.from_uri(state["db_uri"])
        
        if not db:
            return {"result": "Error: Database not available"}
        
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        return {"result": result}
    
    except Exception as e:
        return {"result": f"Error executing query: {str(e)}"}

def generate_answer(state: State) -> dict:
    try:
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question clearly and concisely. "
            "Use markdown formatting for better readability. Use **bold** for important values, "
            "data points, and movie/show titles. Format lists properly with numbers or bullet points "
            "where appropriate.\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}\n\n"
            "Provide a comprehensive answer that directly addresses the question. "
            "Include key numbers and insights from the data."
        )

        response = llm1.invoke(prompt)
        response_content = response.content if hasattr(response, "content") else str(response)
        html_content = markdown.markdown(response_content)
        return {"answer": html_content}
    
    except Exception as e:
        return {"answer": f"<p>Error generating answer: {str(e)}</p>"}