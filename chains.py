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
def build_qa_chain(vector_store, prompt, llm):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
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
    def retrieve_video_profile_docs(video_id):
        docs = vector_store.similarity_search(
            query="What is this video mainly about? What are the key topics discussed?",
            k=10,
            filter={"video_id": video_id}
        )

        return docs

    profile_chain = (
            RunnableLambda(retrieve_video_profile_docs)
            | RunnableLambda(format_docs)
            | prompt
            | llm
            | StrOutputParser()
    )

    return profile_chain

def build_video_similarity_chain(prompt, llm):
    chain = prompt | llm | StrOutputParser()
    return chain