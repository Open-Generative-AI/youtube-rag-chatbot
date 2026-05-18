# YouTube RAG Chatbot

An end-to-end YouTube Retrieval-Augmented Generation chatbot built with **LangChain**, **OpenAI**, **FAISS**, and **Streamlit**.

Users can paste one or more YouTube URLs, process transcripts, ask transcript-grounded questions, summarize videos, and compare multiple videos.

---

## Features

- Paste one or more YouTube links
- Extract video metadata using `yt-dlp`
- Fetch YouTube transcripts
- Create timestamp-aware transcript chunks
- Generate embeddings using OpenAI
- Store and retrieve chunks using FAISS
- Cache FAISS indexes locally for faster repeated usage
- Ask specific questions across uploaded videos
- Summarize each video separately
- Compare multiple videos
- Detect whether videos are similar before generating a detailed comparison
- Handle invalid URLs, unavailable videos, disabled transcripts, and missing English transcripts
- Streamlit-based user interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI
- FAISS
- YouTube Transcript API
- yt-dlp
- python-dotenv

---

## Application Modes

### 1. Ask a Specific Question

Use this mode for targeted questions about uploaded videos.

Example:

```text
What does the video say about generative AI?
What does the Kubernetes video explain about Helm?
How does Virat Kohli perform under pressure?
```

### 2. Summarize Each Video

Use this mode to get an overall summary of every uploaded video.
```text
Example:

Summarize each video.
Explain what each video is about.
Give me the key takeaways from each video.
```

### 3. Compare Videos
```text
Use this mode to compare multiple videos.

The app first generates a short profile for each video, checks whether the videos are similar, and then either:

explains why they are not similar, or
generates a detailed comparison if they are related.
```

## Project Structure
```text
youtube-rag-chatbot/
├── app.py              # Streamlit frontend
├── chatbot.py          # Main chatbot creation workflow
├── chains.py           # LangChain chain builders
├── helpers.py          # URL parsing, transcripts, chunking, FAISS helpers
├── prompts.py          # Prompt templates
├── requirements.txt
├── .env.example
├── .gitignore
├── assets/
│   └── screenshots/
└── vector_cache/       # Local FAISS cache, ignored by git
```

## Setup
### 1. Clone the repository
```text
git clone https://github.com/Open-Generative-AI/youtube-rag-chatbot
cd youtube-rag-chatbot
```

### 2. Create and activate a virtual environment
```text
python -m venv .venv
```

### 3. Windows:
```text
.venv\Scripts\activate
```

### 4. macOS/Linux:
```text
source .venv/bin/activate
```

### 5. Install dependencies
```text
pip install -r requirements.txt
```

### 6. Create .env
```text
Create a .env file in the root folder:
OPENAI_API_KEY=your_openai_api_key_here
```

### 7. Run the app
```text
streamlit run main_app.py
```
## Limitations
The app depends on available YouTube transcripts.
Videos without transcripts cannot be processed.
Visual content from videos is not analyzed.
Timestamp accuracy depends on transcript quality.
FAISS cache is stored locally.

## Future Improvements
Add transcript language selection
Add chat history
Add user authentication
Add database support
Add Docker deployment
Add CI/CD with GitHub Actions
