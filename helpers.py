from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from yt_dlp import YoutubeDL
from langchain_core.documents import Document
import os
import json
import hashlib

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
    errors = []

    for url in urls:
        video_id = get_youtube_id(url)

        if video_id is None:
            errors.append({
                "url": url,
                "error": "Invalid YouTube URL"
            })
            continue

        video_title = get_video_title(url)

        videos.append({
            "video_id": video_id,
            "url": url,
            "video_title": video_title
        })

    return videos, errors

def ingest_document(videos):
    ytt_api = YouTubeTranscriptApi()
    transcripts = {}
    errors = []

    for video in videos:
        video_id = video["video_id"]

        try:
            fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

            transcript_segments = []

            for snippet in fetched_transcript:
                transcript_segments.append({
                    "text": snippet.text,
                    "start": snippet.start,
                    "duration": snippet.duration,
                    "end": snippet.start + snippet.duration
                })

            full_transcript = " ".join(
                segment["text"] for segment in transcript_segments
            )

            if not full_transcript.strip():
                errors.append({
                    "video_id": video_id,
                    "video_title": video["video_title"],
                    "url": video["url"],
                    "error": "Transcript is empty"
                })
                continue

            transcripts[video_id] = {
                "segments": transcript_segments,
                "video_title": video["video_title"],
                "url": video["url"]
            }

        except TranscriptsDisabled:
            errors.append({
                "video_id": video_id,
                "video_title": video["video_title"],
                "url": video["url"],
                "error": "Transcripts are disabled for this video"
            })

        except NoTranscriptFound:
            errors.append({
                "video_id": video_id,
                "video_title": video["video_title"],
                "url": video["url"],
                "error": "No English transcript found for this video"
            })

        except VideoUnavailable:
            errors.append({
                "video_id": video_id,
                "video_title": video["video_title"],
                "url": video["url"],
                "error": "Video is unavailable, private, or removed"
            })

        except Exception as e:
            errors.append({
                "video_id": video_id,
                "video_title": video["video_title"],
                "url": video["url"],
                "error": str(e)
            })

    return transcripts, errors
def chunk_transcripts(transcripts, window_size=120):
    all_chunks = []

    for video_id, video_data in transcripts.items():
        segments = video_data["segments"]
        video_title = video_data["video_title"]
        url = video_data["url"]

        current_text = []
        window_start = None
        window_end = None
        chunk_index = 0

        for segment in segments:
            if window_start is None:
                window_start = segment["start"]

            current_text.append(segment["text"])
            window_end = segment["end"]

            if window_end - window_start >= window_size:
                doc = Document(
                    page_content=" ".join(current_text),
                    metadata={
                        "video_id": video_id,
                        "video_title": video_title,
                        "url": url,
                        "start_time": round(window_start, 2),
                        "end_time": round(window_end, 2),
                        "chunk_index": chunk_index
                    }
                )

                all_chunks.append(doc)

                current_text = []
                window_start = None
                window_end = None
                chunk_index += 1

        if current_text:
            doc = Document(
                page_content=" ".join(current_text),
                metadata={
                    "video_id": video_id,
                    "video_title": video_title,
                    "url": url,
                    "start_time": round(window_start, 2),
                    "end_time": round(window_end, 2),
                    "chunk_index": chunk_index
                }
            )

            all_chunks.append(doc)
    return all_chunks

def get_cache_key(video_ids):
    sorted_ids = sorted(video_ids)
    raw_key = "_".join(sorted_ids)

    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def get_cache_dir(cache_key):
    return os.path.join("vector_cache", cache_key)


def save_vector_store(vector_store, videos, cache_key):
    cache_dir = get_cache_dir(cache_key)

    os.makedirs(cache_dir, exist_ok=True)

    vector_store.save_local(cache_dir)

    metadata_path = os.path.join(cache_dir, "videos.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2)


def load_vector_store(cache_key):
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    cache_dir = get_cache_dir(cache_key)
    metadata_path = os.path.join(cache_dir, "videos.json")

    if not os.path.exists(cache_dir):
        return None, None

    if not os.path.exists(metadata_path):
        return None, None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.load_local(
        cache_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    return vector_store, videos

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store

def format_seconds(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_docs(retrieved_docs):
    formatted_chunks = []

    for doc in retrieved_docs:
        video_id = doc.metadata.get("video_id", "Unknown video_id")
        video_title = doc.metadata.get("video_title", "Unknown title")
        url = doc.metadata.get("url", "Unknown URL")

        start_time = doc.metadata.get("start_time")
        end_time = doc.metadata.get("end_time")

        if start_time is not None and end_time is not None:
            timestamp = f"{format_seconds(start_time)} - {format_seconds(end_time)}"
            timestamped_url = f"{url}&t={int(start_time)}s"
        else:
            timestamp = "Unknown timestamp"
            timestamped_url = url

        formatted_chunks.append(
            f"""
Video Title: {video_title}
Video ID: {video_id}
Timestamp: {timestamp}
Video URL: {timestamped_url}

Transcript Chunk:
{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(formatted_chunks)
