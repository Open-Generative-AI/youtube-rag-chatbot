import streamlit as st
from dotenv import load_dotenv

from chatbot import create_youtube_chatbot

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 YouTube RAG Chatbot")
st.write("Paste one or more YouTube links and chat with the video transcripts.")

# -----------------------------
# Session state initialization
# -----------------------------
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "video_ids" not in st.session_state:
    st.session_state.video_ids = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# URL input section
# -----------------------------
st.subheader("1. Add YouTube Links")

urls_text = st.text_area(
    "Paste YouTube URLs, one per line",
    height=150,
    placeholder="""https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2"""
)

process_button = st.button("Process Videos")


if process_button:
    urls = [
        url.strip()
        for url in urls_text.splitlines()
        if url.strip()
    ]

    if not urls:
        st.error("Please paste at least one YouTube URL.")
    else:
        with st.spinner("Processing videos... fetching transcripts, chunking, embedding, and creating vector store."):
            chatbot = create_youtube_chatbot(urls)

            st.session_state.chatbot = chatbot
            st.session_state.video_ids = chatbot["video_ids"]
            st.session_state.messages = []

        st.success("Videos processed successfully.")


# -----------------------------
# Show loaded videos
# -----------------------------
if st.session_state.chatbot is None:
    st.warning("No videos processed yet. Paste real YouTube URLs and click Process Videos.")

if st.session_state.chatbot is not None:
    st.subheader("2. Loaded Videos")

    for video_id in st.session_state.video_ids:
        st.write(f"- `{video_id}`")

    st.subheader("3. Choose Chat Mode")

    mode = st.radio(
        "Select how you want to chat",
        [
            "Ask across all videos",
            "Ask one selected video"
        ]
    )

    selected_video_id = None

    if mode == "Ask one selected video":
        selected_video_id = st.selectbox(
            "Select a video",
            st.session_state.video_ids
        )

    st.subheader("4. Chat with your video(s)")
    col1, col2 = st.columns([1, 5])

    with col1:
        clear_chat = st.button("🗑️ Clear Conversation")

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask a question about the video(s)...")

    if user_question:
        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                if mode == "Ask across all videos":
                    response = st.session_state.chatbot["all_video_chain"].invoke(
                        user_question
                    )

                elif mode == "Ask one selected video":
                    response = st.session_state.chatbot["selected_video_chain"].invoke({
                        "question": user_question,
                        "video_id": selected_video_id
                    })

                st.write(response)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })