from flask import Flask, request, jsonify, render_template
import sys
import os
import markdown
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.youtube_video_summary import extract_video_id, generate_summary
from src.features.chat_with_website import prepare_website_data, chat_with_website
from src.features.text_to_sql import write_query, execute_query, generate_answer
from src.features.chat_with_documents import chat_with_document, load_document, clear_documents
from src.features.chat_with_documents import speech_to_text

from config import MODELS, get_model_display_name
from config import DATABASE_URI

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)  

# ============= Serves the main application homepage with available models ===============
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=MODELS)

# ============= Generates summaries from YouTube videos using AI models ===============
@app.route('/api/youtube-summary', methods=['POST'])
def youtube_summary_api():
    data = request.get_json()
    youtube_url = data['youtube_url']
    model = data.get('model', 'llama') 
    
    video_id = extract_video_id(youtube_url)
    summary = generate_summary(youtube_url, model)
    processed_summary = markdown.markdown(summary)
    
    return jsonify({
        'summary': processed_summary,
        'video_id': video_id,
        'model': model
    })

# ============= Crawls and processes website content for later Q&A ===============
@app.route('/api/prepare-website', methods=['POST'])
def prepare_website_api():
    data = request.get_json()
    website_url = data['website_url']
    
    result = prepare_website_data(website_url)
    
    return jsonify({
        'success': result['success'],
        'message': result.get('message', ''),
        'website_url': website_url
    })

# ============= Answers questions about previously processed website content ===============
@app.route('/api/chat-with-website', methods=['POST'])
def chat_with_website_api():
    data = request.get_json()
    website_url = data['website_url']
    question = data['question']
    model_name = data.get('model', 'deepseek')
    
    raw_answer = chat_with_website(website_url, question, model_name)
    
    return jsonify({
        'website_url': website_url,
        'question': question,
        'answer': markdown.markdown(raw_answer),
        'model_used': get_model_display_name(model_name)
    })

# ============= Translates natural language to SQL, executes queries, and returns results ===============
@app.route("/api/text_to_sql", methods=["POST"])
def text_to_sql_api():
    state = {"question": "", "query": "", "result": "", "answer": "", "db_uri": DATABASE_URI}
    
    state["question"] = request.form.get("question", "").strip()
    if state["question"]:
        query_dict = write_query(state)
        state["query"] = query_dict.get("query", "")

        result_dict = execute_query(state)
        state["result"] = result_dict.get("result", "")

        answer_dict = generate_answer(state)
        state["answer"] = answer_dict.get("answer", "")

    return jsonify(state)

# ============= Uploads and processes documents for later Q&A interactions ===============
@app.route('/api/upload-document', methods=['POST'])
def upload_document_api():
    file = request.files['document']
    
    upload_dir = Path(__file__).parent / 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = upload_dir / file.filename
    file.save(file_path)
    
    result = load_document(str(file_path))
    
    return jsonify({
        'success': result['success'],
        'message': 'Document uploaded and processed successfully',
        'document_name': file.filename
    })

# ============= Answers questions about previously uploaded documents ===============
@app.route('/api/chat-with-document', methods=['POST'])
def chat_with_document_api():
    data = request.get_json()
    question = data['question']
    model_name = data.get('model', 'deepseek')
    
    raw_answer = chat_with_document(question, model_name)
    
    return jsonify({
        'question': question,
        'answer': markdown.markdown(raw_answer),
        'model_used': get_model_display_name(model_name)
    })

# ============= Removes all uploaded documents from memory ===============
@app.route('/api/clear-documents', methods=['POST'])
def clear_documents_api():
    result = clear_documents()
    
    return jsonify({
        'success': result['success'],
        'message': 'Documents cleared successfully'
    })

# ============= Converts audio recordings to text transcripts ===============
@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text_api():
    file = request.files['audio']
    
    upload_dir = Path(__file__).parent / 'uploads' / 'temp'
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = upload_dir / file.filename
    file.save(file_path)
    
    transcript = speech_to_text(str(file_path))
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return jsonify({
        'success': True,
        'transcript': transcript
    })

# ============= Retrieves database schema information for SQL generation ===============
@app.route("/api/database_tables", methods=["GET"])
def database_tables_api():
    from src.features.text_to_sql import SQLDatabase
    
    db = SQLDatabase.from_uri(DATABASE_URI)
    table_info = db.get_table_info()
    
    import re
    tables = re.findall(r'CREATE TABLE (\w+)', table_info)
    
    return jsonify({
        "success": True,
        "tables": tables,
        "table_info": table_info
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)