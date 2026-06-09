import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

print("LANGSMITH_TRACING:", os.getenv("LANGSMITH_TRACING"))
print("LANGSMITH_ENDPOINT:", os.getenv("LANGSMITH_ENDPOINT"))
print("LANGSMITH_API_KEY:", os.getenv("LANGSMITH_API_KEY"))
print("LANGSMITH_PROJECT:", os.getenv("LANGSMITH_PROJECT"))

try:
    client = Client()
    # List projects to verify authentication
    projects = list(client.list_projects())
    print("\n[SUCCESS] Connected to LangSmith!")
    print("Projects:", [p.name for p in projects])
except Exception as e:
    print("\n[FAILED] Connection failed:")
    print(e)
