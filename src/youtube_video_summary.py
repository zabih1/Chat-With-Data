import os , sys
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.prompt import youtube_video_summary_template
from app.llm import llama_model


def get_video_transcript(video_id):

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    
    full_transcript = ""
    for segment in transcript:
        full_transcript += segment.text + " "
    return full_transcript


def generate_summary(youtube_video_url):

    video_id = youtube_video_url.split("youtu.be/")[1].split("?")[0]

    full_transcript = get_video_transcript(video_id)
    
    prompt_template = PromptTemplate.from_template(youtube_video_summary_template)
    prompt = prompt_template.format(full_transcript=full_transcript)

    llm = llama_model()
    summary = llm.invoke(prompt)
    
    return summary.content




response = generate_summary("https://youtu.be/T-D1OfcDW1M?si=g-jIb0p49KZkSVPb")

print(response)

