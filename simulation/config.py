import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

APP_ENV = os.getenv("APP_ENV", "development")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")