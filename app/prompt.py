chat_with_document_template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    
    Question: {question} 
    Context: {context} 

    Answer:
    
    """

chat_with_website_template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    
    Question: {question} 
    Context: {context} 
    

    Answer:
    
    """

youtube_video_summary_template = """
# Role: YouTube Video Summarizer

## Task
Provide a concise, informative summary of the YouTube video transcript below.

## Guidelines
- Focus on the main ideas, key points, and essential takeaways
- Organize information in a structured, easy-to-read format with bullet points
- Include any significant conclusions or insights presented
- Keep the summary under 250 words
- Exclude unnecessary details or tangential information

## Transcript

{full_transcript}

## Output Format
- Title: [Inferred title based on content]
- Main Topic: [1-2 sentence description of what the video is about]
- Key Points:
  • [Point 1]
  • [Point 2]
  • [Point 3]
  • [Additional points as needed]
- Conclusion: [Main takeaway or call to action]
"""


text_to_sql_templates = """  

Given an input question, generate a syntactically correct `{dialect}` query to retrieve relevant results. If the user does not specify the number of examples, limit the query to `{top_k}` results. Order the results by a relevant column to return the most meaningful examples.  

Only select the necessary columns based on the question—avoid querying all columns from a table.  

Use only the column names available in the schema and ensure correctness when referencing tables and columns.  

Available tables:  
`{table_info}`  

**Question:** `{input}`  

"""  