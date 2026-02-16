# env.py
import os
from dotenv import load_dotenv

# Load once, early
load_dotenv()

def get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val