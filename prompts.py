from langchain_core.prompts import PromptTemplate


all_video_prompt = PromptTemplate(
    template="""
You are a helpful YouTube video assistant.

Answer the user's question using only the transcript context below.

The context may contain transcript chunks from one or more YouTube videos.
Each chunk includes a Video ID.

If you use information from a video, mention the relevant Video ID.

If the answer is not present in the context, say:
"I could not find this information in the video transcript."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


selected_video_prompt = PromptTemplate(
    template="""
You are a helpful YouTube video assistant.

Answer the user's question using only the selected video's transcript context.

The user selected this video_id:
{video_id}

If the context contains relevant information, answer clearly.

If the context only partially answers the question, say what you found and mention that the transcript context may not contain the full answer.

Only if the context has no relevant information at all, say:
"I could not find this information in the selected video transcript."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question", "video_id"]
)


video_profile_prompt = PromptTemplate(
    template="""
You are analyzing a YouTube video using its title and transcript.

Use ONLY the provided context.
Do not guess topics that are not clearly supported by the title or transcript.

The context contains:
- Video Title
- Video ID
- Video URL
- Transcript Chunk

Create a short profile of the video.

Return the answer in this exact format:

Video title:
...

Main topic:
...

Key ideas:
- ...
- ...
- ...

One-sentence summary:
...

If the transcript is short or incomplete, say:
"This profile is based on the available title and transcript."

Context:
{context}

Video profile:
""",
    input_variables=["context"]
)

video_similarity_prompt = PromptTemplate(
    template="""
You are checking whether multiple YouTube videos are similar enough to compare.

You will receive short profiles for each video.

Decide whether the videos are about the same topic or closely related topics.

Return your answer in this exact format:

Similarity: YES or NO

Reason:
...

Video-by-video explanation:
- Video ID: ...
  Topic: ...
- Video ID: ...
  Topic: ...

Profiles:
{profiles}

Answer:
""",
    input_variables=["profiles"]
)

selected_video_summary_prompt = PromptTemplate(
    template="""
You are a helpful YouTube video summarizer.

Summarize the selected YouTube video using only the transcript context below.

The context includes the video title, video ID, video URL, and transcript chunks.

Create a clear, useful summary.

Include:
1. Main topic
2. Important points
3. Simple layman explanation
4. Key takeaway

If the transcript context is too limited, say that the summary is based only on the available transcript.

Context:
{context}

Summary:
""",
    input_variables=["context"]
)