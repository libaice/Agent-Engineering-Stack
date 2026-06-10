import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

# Verify that environment variables are loaded
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

print("LANGFUSE_PUBLIC_KEY:", public_key[:10] + "..." if public_key else None)
print("LANGFUSE_SECRET_KEY:", secret_key[:10] + "..." if secret_key else None)
print("LANGFUSE_HOST:", host)

if not public_key or not secret_key:
    print("\n[ERROR] Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY in environment variables.")
    exit(1)

try:
    print("\nConnecting to Langfuse...")
    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )
    
    # Test project authentication by fetching project details
    auth_check = langfuse.auth_check()
    if auth_check:
        print("[SUCCESS] Successfully connected to Langfuse locally!")
        
        # Test creating a trace via start_observation
        trace = langfuse.start_observation(
            name="test-connection-trace",
            input="Ping",
            output="Pong"
        )
        print(f"[SUCCESS] Test trace created with ID: {trace.id}")
        
        # Flush to send events
        langfuse.flush()
        print("[SUCCESS] Event flushed successfully.")
    else:
        print("[FAILED] Connection check failed. Please check your credentials.")
except Exception as e:
    print(f"[ERROR] Exception during connection test: {e}")
