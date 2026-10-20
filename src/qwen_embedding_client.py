import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")









def init_qwen_embedding_client(api_key=None,base_url=None):
    client = OpenAI(api_key=api_key,base_url=base_url)
    return client