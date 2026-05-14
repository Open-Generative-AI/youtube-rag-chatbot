from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from helpers import format_docs

def format_video_profiles(video_profiles):
    formatted_profiles = []

    for video_id, profile in video_profiles.items():
        formatted_profiles.append(
            f"Video ID: {video_id}\n{profile}"
        )

    return "\n\n---\n\n".join(formatted_profiles)
def build_qa_chain(vector_store, prompt, llm, video_ids, k_per_video=4):

    def retrieve_docs_from_all_videos(question):
        all_docs = []

        for video_id in video_ids:
            docs = vector_store.similarity_search(
                query=question,
                k=k_per_video,
                filter={"video_id": video_id}
            )

            all_docs.extend(docs)

        return all_docs

    parallel_chain = RunnableParallel({
        "context": RunnableLambda(retrieve_docs_from_all_videos) | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    chain = parallel_chain | prompt | llm | parser

    return chain

def build_selected_video_qa_chain(vector_store, prompt, llm):
    def retrieve_selected_video_docs(inputs):
        question = inputs["question"]
        video_id = inputs["video_id"]

        docs = vector_store.similarity_search(
            query=question,
            k=8,
            filter={"video_id": video_id}
        )

        return docs

    selected_video_chain = RunnableParallel({
        "context": RunnableLambda(retrieve_selected_video_docs) | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda inputs: inputs["question"]),
        "video_id": RunnableLambda(lambda inputs: inputs["video_id"])
    })

    parser = StrOutputParser()

    chain = selected_video_chain | prompt | llm | parser

    return chain


def build_video_profile_chain(vector_store, prompt, llm):

    def get_video_docs(video_id):
        docs = []

        for docstore_id in vector_store.index_to_docstore_id.values():
            doc = vector_store.docstore.search(docstore_id)

            if doc.metadata.get("video_id") == video_id:
                docs.append(doc)

        # If video has many chunks, take a representative sample:
        # first 5, middle 5, last 5
        if len(docs) > 15:
            first_docs = docs[:5]

            middle_start = max(5, len(docs) // 2 - 2)
            middle_docs = docs[middle_start:middle_start + 5]

            last_docs = docs[-5:]

            docs = first_docs + middle_docs + last_docs

        return docs

    profile_chain = (
        RunnableLambda(get_video_docs)
        | RunnableLambda(format_docs)
        | prompt
        | llm
        | StrOutputParser()
    )

    return profile_chain
def build_video_similarity_chain(prompt, llm):
    chain = prompt | llm | StrOutputParser()
    return chain

def build_selected_video_summary_chain(vector_store, prompt, llm):

    def retrieve_selected_video_docs(video_id):
        docs = vector_store.similarity_search(
            query="Summarize this video. What are the main topic, important points, examples, and key takeaway?",
            k=12,
            filter={"video_id": video_id}
        )

        return docs

    chain = (
        RunnableLambda(retrieve_selected_video_docs)
        | RunnableLambda(format_docs)
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain