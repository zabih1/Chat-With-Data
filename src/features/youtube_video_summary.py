from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
import sys
import os
import re
from youtube_transcript_api.proxies import WebshareProxyConfig


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.prompts import YOUTUBE_SUMMARY_TEMPLATE
from src.core.llm import get_llm

def get_video_transcript(video_id):

    # ytt_api = YouTubeTranscriptApi()

    ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username="zdvzltzb",
        proxy_password="by7xjzhcw471",
    ))
    

    transcript = ytt_api.fetch(video_id)
    
    full_transcript = ""
    for segment in transcript:
        full_transcript += segment.text + " "
    return full_transcript

def extract_video_id(youtube_url):
    if 'youtu.be/' in youtube_url:
        return youtube_url.split('youtu.be/')[1].split('?')[0]
    
    if 'youtube.com/watch' in youtube_url:
        match = re.search(r'v=([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            return match.group(1)
    
    return youtube_url.split("youtu.be/")[1].split("?")[0]

def generate_summary(youtube_url, model_name='llama'):
    video_id = extract_video_id(youtube_url)
    full_transcript = get_video_transcript(video_id)
    
    prompt_template = PromptTemplate.from_template(YOUTUBE_SUMMARY_TEMPLATE)
    prompt = prompt_template.format(full_transcript=full_transcript)

    llm = get_llm(model_name)
    
    summary = llm.invoke(prompt)
    return summary.content