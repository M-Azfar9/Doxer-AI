from langchain_google_genai import ChatGoogleGenerativeAI
import os 
judge = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key="",
    temperature=0.0,
    max_retries=3,
)

result = judge.invoke("Who are you?")

print(result)