# YouTube RAG Chatbot

An end-to-end YouTube RAG chatbot built with LangChain, OpenAI embeddings, FAISS, and Streamlit.

## Features

- Paste one or more YouTube URLs
- Extract transcripts automatically
- Generate embeddings using OpenAI
- Store chunks in FAISS
- Ask questions across videos
- Summarize videos
- Compare videos using generated video profiles
- Streamlit frontend

## Tech Stack

- Python
- LangChain
- OpenAI
- FAISS
- YouTube Transcript API
- yt-dlp
- Streamlit

## Current Architecture

YouTube URLs → Video metadata → Transcript → Chunks → Embeddings → FAISS → Retriever → LLM → Streamlit UI