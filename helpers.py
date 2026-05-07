from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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

def extract_url_ids(urls):
    url_ids = []

    for url in urls:
        video_id = get_youtube_id(url)

        if video_id is not None:
            url_ids.append(video_id)

    return url_ids

def ingest_document(url_ids):
    ytt_api = YouTubeTranscriptApi()
    transcripts = {}
    for video_id in url_ids:
        try:
            fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

            transcript = " ".join(
                snippet.text for snippet in fetched_transcript
            )
            transcripts[video_id] = transcript
        except TranscriptsDisabled:
            print(f"No captions available for video_id: {video_id}")
    return transcripts

def chunk_transcripts(transcripts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []

    for video_id, transcript in transcripts.items():
        chunks = splitter.create_documents(
            texts=[transcript],
            metadatas=[{"video_id": video_id}]
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
        video_id = doc.metadata["video_id"]

        formatted_chunks.append(
            f"Video ID: {video_id}\nTranscript Chunk:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted_chunks)

