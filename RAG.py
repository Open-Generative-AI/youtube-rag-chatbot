from dotenv import load_dotenv
from chatbot import create_youtube_chatbot
from chains import format_video_profiles
load_dotenv()


urls = ["https://www.youtube.com/watch?v=ZHCB09O6zUk",
        "https://www.youtube.com/watch?v=qYNweeDHiyU"]

chatbot = create_youtube_chatbot(urls)

# print("Loaded video IDs:")
# print(chatbot["video_ids"])
#
#
# print("\n--- All Video QA ---")
# response = chatbot["all_video_chain"].invoke(
#     "What are the main topics discussed across these videos?"
# )
# print(response)
#
#
# print("\n--- Selected Video QA ---")
#
# selected_video_id = chatbot["video_ids"][0]
#
# user_question = input("Ask a question about the selected video: ")
#
# response = chatbot["selected_video_chain"].invoke({
#     "question": user_question,
#     "video_id": selected_video_id
# })
#
# print(response)
#
# video_profiles = {}
#
# for video_id in chatbot["video_ids"]:
#     profile = chatbot["video_profile_chain"].invoke(video_id)
#     video_profiles[video_id] = profile
#
# profiles_text = format_video_profiles(video_profiles)
#
# similarity_result = chatbot["video_similarity_chain"].invoke({
#     "profiles": profiles_text
# })

# print("\n--- Selected Video Summary ---")
#
# selected_video_id = chatbot["video_ids"][0]
#
# summary = chatbot["selected_video_summary_chain"].invoke(
#     selected_video_id
# )
#
# print(summary)

# print(similarity_result)