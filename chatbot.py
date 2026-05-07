from langchain_openai import ChatOpenAI
from helpers import extract_url_ids, ingest_document, chunk_transcripts, create_vector_store
from chains import (build_qa_chain, build_selected_video_qa_chain, build_video_profile_chain, build_video_similarity_chain, format_video_profiles)
from prompts import (all_video_prompt, selected_video_prompt, video_profile_prompt, video_similarity_prompt)
def create_youtube_chatbot(urls):
    url_ids = extract_url_ids(urls)

    transcripts = ingest_document(url_ids)

    chunks = chunk_transcripts(transcripts)

    vector_store = create_vector_store(chunks)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    all_video_chain = build_qa_chain(
        vector_store=vector_store,
        prompt=all_video_prompt,
        llm=llm
    )

    selected_video_chain = build_selected_video_qa_chain(
        vector_store=vector_store,
        prompt=selected_video_prompt,
        llm=llm
    )

    video_profile_chain = build_video_profile_chain(
        vector_store=vector_store,
        prompt=video_profile_prompt,
        llm=llm
    )

    video_similarity_chain = build_video_similarity_chain(
        prompt=video_similarity_prompt,
        llm=llm
    )

    return {
        "video_ids": url_ids,
        "all_video_chain": all_video_chain,
        "selected_video_chain": selected_video_chain,
        "video_profile_chain": video_profile_chain,
        "video_similarity_chain": video_similarity_chain,
        "vector_store": vector_store
    }