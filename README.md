# Chat-With-Data

A comprehensive application for interacting with various data sources using conversational AI.


📽️ [Watch the demo video on Google Drive](https://drive.google.com/file/d/1j7f7FHw9Gs2nieY6tzm4I5w9xDWKBIIw/view?usp=sharing)

## Features

- **Chat with Documents** - Upload and chat with your documents (PDF, DOCX, TXT, etc.)
- **Chat with Websites** - Extract and chat with website content
- **YouTube Video Summarizer** - Get AI-generated summaries of YouTube videos
- **Text to SQL** - Query databases using natural language

## Models

The application supports multiple AI models:
- Llama 3
- Gemini
- Mistral
- DeepSeek

## Tech Stack

- **Backend**: Flask, Python
- **Frontend**: HTML, Bootstrap, JavaScript
- **AI/ML**: LangChain, Google Generative AI, Groq
- **Other**: YouTube Transcript API, Llama Parse API for document processing

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - GEMINI_API_KEY
   - GROQ_API_KEY
   - LLAMA_PARSER_API
   - DATABASE_URI

4. Run the application:
   ```
   python run.py
   ```

## Usage

Access the web interface at http://localhost:5000 and use the sidebar to navigate between features:

- Upload documents to chat with their content
- Enter website URLs to chat about website information
- Paste YouTube video links to generate summaries
- Ask natural language questions to query your database

