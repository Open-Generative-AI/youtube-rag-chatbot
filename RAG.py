from dotenv import load_dotenv
from chatbot import create_youtube_chatbot
from chains import format_video_profiles
load_dotenv()


urls = [
    "https://www.youtube.com/watch?v=LPZh9BOjkQs",
    "https://www.youtube.com/watch?v=X48VuDVv0do&t=204s"
]

chatbot = create_youtube_chatbot(urls)

print("Loaded video IDs:")
print(chatbot["video_ids"])


print("\n--- All Video QA ---")
response = chatbot["all_video_chain"].invoke(
    "What are the main topics discussed across these videos?"
)
print(response)


print("\n--- Selected Video QA ---")
response = chatbot["selected_video_chain"].invoke({
    "question": "What does the video say about transformers?",
    "video_id": "LPZh9BOjkQs"
})
print(response)

video_profiles = {}

for video_id in chatbot["video_ids"]:
    profile = chatbot["video_profile_chain"].invoke(video_id)
    video_profiles[video_id] = profile

profiles_text = format_video_profiles(video_profiles)

similarity_result = chatbot["video_similarity_chain"].invoke({
    "profiles": profiles_text
})

print(similarity_result)