import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

print("API key loaded:", api_key is not None)
print("Base URL:", base_url)
print("Model:", model)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Reply with only: API working"}
    ],
    temperature=0,
)

print(response.choices[0].message.content)