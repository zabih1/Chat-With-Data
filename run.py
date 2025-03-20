from flask import Flask, request, jsonify, render_template
import sys
import os
import markdown

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.youtube_video_summary import extract_video_id, generate_summary
from src.chat_with_website import scrape_website_content, save_website_content, load_website_content, create_rag_chain
from app.utils import process_markdown, MODELS, get_model_display_name

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', models=MODELS)


@app.route('/chat-with-website', methods=['GET'])
def chat_with_website_page():
    return render_template('chat_with_website.html')


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
        
        return jsonify({
            'summary': summary,
            'video_id': video_id,
            'model': model
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat-with-website', methods=['POST'])
def chat_with_website_api():
    try:
        data = request.get_json()
        
        if not data or 'website_url' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required parameters: website_url and/or question'}), 400
        
        website_url = data['website_url']
        question = data['question']
        
        # Check if we have this website cached or need to load it
        # For simplicity, we'll just load it every time in this implementation
        documents = load_website_content(website_url)
        
        # Create RAG chain
        rag_chain = create_rag_chain(documents)
        
        # Get answer
        answer = rag_chain.invoke(question)
        
        return jsonify({
            'website_url': website_url,
            'question': question,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)