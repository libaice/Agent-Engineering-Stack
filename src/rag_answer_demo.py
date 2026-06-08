import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from pdf_chunk_demo import chunk_pdf
from vector_search_demo import SimpleVectorStore



load_dotenv()




client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, who are you , and what is your training time ?"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)