import streamlit as st
from dotenv import load_dotenv

from chatbot import create_youtube_chatbot
from chains import format_video_profiles

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

if "videos" not in st.session_state:
    st.session_state.videos = []

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
            st.session_state.videos = chatbot["videos"]
            st.session_state.messages = []

        st.success("Videos processed successfully.")


# -----------------------------
# Empty state
# -----------------------------
if st.session_state.chatbot is None:
    st.warning("No videos processed yet. Paste real YouTube URLs and click Process Videos.")


# -----------------------------
# Main app after videos are loaded
# -----------------------------
if st.session_state.chatbot is not None:
    st.subheader("2. Loaded Videos")

    for i, video in enumerate(st.session_state.videos, start=1):
        st.write(f"{i}. **{video['video_title']}**")

    st.subheader("3. Choose Chat Mode")

    mode = st.radio(
        "Select what you want to do",
        [
            "Ask a specific question",
            "Summarize each video",
            "Compare videos"
        ]
    )

    if mode == "Ask a specific question":
        st.info(
            "Use this mode for targeted questions about specific information in the uploaded video(s). "
            "Examples: 'What does the video say about Virat Kohli?', "
            "'What does the Kubernetes video explain about Helm?', "
            "'How does the speaker explain transformers?'"
        )

    elif mode == "Summarize each video":
        st.info(
            "Use this mode when you want an overall explanation or summary of every uploaded video. "
            "Examples: 'Summarize each video', 'Explain what each video is about', "
            "'Give me the key takeaways from each video.'"
        )

    elif mode == "Compare videos":
        st.info(
            "Use this mode to check whether the uploaded videos are related and compare their topics. "
            "Example: 'Compare these videos and explain whether they are similar.'"
        )

    st.subheader("4. Chat with your video(s)")

    col1, col2 = st.columns([1, 5])

    with col1:
        clear_chat = st.button("🗑️ Clear Conversation")

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Type your question based on the selected mode...")

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

                if mode == "Ask a specific question":
                    response = st.session_state.chatbot["all_video_chain"].invoke(
                        user_question
                    )

                elif mode == "Summarize each video":
                    summaries = []

                    for video in st.session_state.videos:
                        video_id = video["video_id"]
                        title = video["video_title"]

                        summary = st.session_state.chatbot["selected_video_summary_chain"].invoke(
                            video_id
                        )

                        summaries.append(
                            f"### {title}\n\n{summary}"
                        )

                    response = "\n\n---\n\n".join(summaries)

                elif mode == "Compare videos":
                    if len(st.session_state.videos) < 2:
                        response = "Please paste at least two YouTube videos to compare."
                    else:
                        video_profiles = {}

                        for video in st.session_state.videos:
                            video_id = video["video_id"]
                            profile = st.session_state.chatbot["video_profile_chain"].invoke(
                                video_id
                            )
                            video_profiles[video_id] = profile

                        profiles_text = format_video_profiles(video_profiles)

                        with st.expander("View generated video profiles"):
                            st.write(profiles_text)

                        response = st.session_state.chatbot["video_similarity_chain"].invoke({
                            "profiles": profiles_text
                        })

                st.write(response)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })