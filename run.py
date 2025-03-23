from flask import Flask, request, jsonify, render_template
import sys
import os
import markdown
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.youtube_video_summary import extract_video_id, generate_summary
from src.chat_with_website import prepare_website_data, chat_with_website
from src.text_to_sql import write_query, execute_query, generate_answer
from app.utils import  MODELS, get_model_display_name
from config import db_uri

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  

@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html', models=MODELS)

@app.route('/api/youtube-summary', methods=['POST'])
def youtube_summary_api():
    try:
        data = request.get_json()
        
        if not data or 'youtube_url' not in data:
            return jsonify({'error': 'Missing required parameter: youtube_url'}), 400
        
        youtube_url = data['youtube_url']
        model = data.get('model', 'llama') 
        
        if 'youtu.be/' not in youtube_url and 'youtube.com/' not in youtube_url:
            return jsonify({'error': 'Invalid YouTube URL format'}), 400
        
        video_id = extract_video_id(youtube_url)
        summary = generate_summary(youtube_url, model)
        
        processed_summary = markdown.markdown(summary)
        
        return jsonify({
            'summary': processed_summary,
            'video_id': video_id,
            'model': model
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prepare-website', methods=['POST'])
def prepare_website_api():
    try:
        data = request.get_json()
        
        if not data or 'website_url' not in data:
            return jsonify({'error': 'Missing required parameter: website_url'}), 400
        
        website_url = data['website_url']
        
        result = prepare_website_data(website_url)
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Unknown error')}), 500
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'website_url': website_url
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat-with-website', methods=['POST'])
def chat_with_website_api():
    try:
        data = request.get_json()
        
        if not data or 'website_url' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required parameters: website_url and/or question'}), 400
        
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
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/text_to_sql", methods=["POST"])
def text_to_sql_api():
    state = {"question": "", "query": "", "result": "", "answer": "", "db_uri": db_uri}

    if request.method == "POST":
        state["question"] = request.form.get("question", "").strip()
        if state["question"]:
            try:
                query_dict = write_query(state)
                state["query"] = query_dict.get("query", "")

                result_dict = execute_query(state)
                state["result"] = result_dict.get("result", "")

                answer_dict = generate_answer(state)
                state["answer"] = answer_dict.get("answer", "")

                return jsonify(state)
            
            except Exception as e:
                return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify(state)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)