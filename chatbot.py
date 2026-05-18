from langchain_openai import ChatOpenAI
from helpers import (extract_video_metadata, ingest_document, chunk_transcripts, create_vector_store, get_cache_key, load_vector_store, save_vector_store)
from chains import (build_qa_chain, build_selected_video_qa_chain, build_video_profile_chain, build_video_similarity_chain, format_video_profiles, build_selected_video_summary_chain, build_video_comparison_chain)
from prompts import (all_video_prompt, selected_video_prompt, video_profile_prompt, video_similarity_prompt, selected_video_summary_prompt,video_comparison_prompt)

def create_youtube_chatbot(urls):
    videos, metadata_errors = extract_video_metadata(urls)
    transcripts, transcript_errors = ingest_document(videos)
    processed_video_ids = list(transcripts.keys())
    processed_videos = [
        video for video in videos
        if video["video_id"] in processed_video_ids
    ]
    url_ids = processed_video_ids
    errors = metadata_errors + transcript_errors
    cache_key = get_cache_key(url_ids)
    cache_used = False
    vector_store, cached_videos = load_vector_store(cache_key)

    if vector_store is None:
        chunks = chunk_transcripts(transcripts)

        vector_store = create_vector_store(chunks)

        save_vector_store(
            vector_store=vector_store,
            videos=processed_videos,
            cache_key=cache_key
        )
    else:
        processed_videos = cached_videos
        cache_used = True

    if not transcripts:
        return {
            "videos": [],
            "video_ids": [],
            "errors": errors,
            "all_video_chain": None,
            "selected_video_chain": None,
            "selected_video_summary_chain": None,
            "video_profile_chain": None,
            "video_similarity_chain": None,
            "vector_store": None
        }

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    all_video_chain = build_qa_chain(
        vector_store=vector_store,
        prompt=all_video_prompt,
        llm=llm,
        video_ids=url_ids
    )

    selected_video_chain = build_selected_video_qa_chain(
        vector_store=vector_store,
        prompt=selected_video_prompt,
        llm=llm
    )

    selected_video_summary_chain = build_selected_video_summary_chain(
        vector_store=vector_store,
        prompt=selected_video_summary_prompt,
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

    video_comparison_chain = build_video_comparison_chain(
        prompt=video_comparison_prompt,
        llm=llm
    )

    return {
        "videos": processed_videos,
        "video_ids": url_ids,
        "errors": errors,
        "cache_used": cache_used,
        "all_video_chain": all_video_chain,
        "selected_video_chain": selected_video_chain,
        "selected_video_summary_chain": selected_video_summary_chain,
        "video_profile_chain": video_profile_chain,
        "video_similarity_chain": video_similarity_chain,
        "video_comparison_chain": video_comparison_chain,
        "vector_store": vector_store
    }