from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from yt_dlp import YoutubeDL


def get_video_title(url):
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "Unknown Title")

    except Exception:
        return "Unknown Title"
def get_youtube_id(url):
    parsed_url = urlparse(url)

    # Handle shortened youtu.be URLs
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]

    # Handle standard youtube.com URLs
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        # Standard watch URLs: /watch?v=VIDEO_ID
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            return query.get('v', [None])[0]

        # Embed URLs: /embed/VIDEO_ID or /v/VIDEO_ID
        path_parts = parsed_url.path.split('/')
        if len(path_parts) > 2 and path_parts[1] in ('embed', 'v'):
            return path_parts[2]
    return None

def extract_video_metadata(urls):
    videos = []

    for url in urls:
        video_id = get_youtube_id(url)

        if video_id is not None:
            videos.append({
                "video_id": video_id,
                "url": url,
                "video_title": get_video_title(url)
            })

    return videos

def ingest_document(videos):
    ytt_api = YouTubeTranscriptApi()
    transcripts = {}

    for video in videos:
        video_id = video["video_id"]

        try:
            fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

            transcript = " ".join(
                snippet.text for snippet in fetched_transcript
            )

            transcripts[video_id] = {
                "transcript": transcript,
                "video_title": video["video_title"],
                "url": video["url"]
            }

        except TranscriptsDisabled:
            print(f"No captions available for video_id: {video_id}")

    return transcripts

def chunk_transcripts(transcripts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []

    for video_id, video_data in transcripts.items():
        transcript = video_data["transcript"]
        video_title = video_data["video_title"]
        url = video_data["url"]

        chunks = splitter.create_documents(
            texts=[transcript],
            metadatas=[{
                "video_id": video_id,
                "video_title": video_title,
                "url": url
            }]
        )

        all_chunks.extend(chunks)

    return all_chunks


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store

def format_docs(retrieved_docs):
    formatted_chunks = []

    for doc in retrieved_docs:
        video_id = doc.metadata.get("video_id", "Unknown video_id")
        video_title = doc.metadata.get("video_title", "Unknown title")
        url = doc.metadata.get("url", "Unknown URL")

        formatted_chunks.append(
            f"""
Video Title: {video_title}
Video ID: {video_id}
Video URL: {url}

Transcript Chunk:
{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(formatted_chunks)
