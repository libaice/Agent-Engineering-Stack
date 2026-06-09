from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import (

    SimpleDirectoryReader,

    VectorStoreIndex,

    Settings,

)

from llama_index.llms.openai import OpenAI

from llama_index.embeddings.openai import OpenAIEmbedding
