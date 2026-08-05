import os

from dotenv import load_dotenv

load_dotenv()


def get_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value
