from flask import Flask, request, jsonify, render_template
import sys
import os
import markdown

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.youtube_video_summary import extract_video_id, generate_summary
from app.utils import process_markdown, MODELS, get_model_display_name

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', models=MODELS)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)