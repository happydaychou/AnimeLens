from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

# Load environment variables from a local .env file when present.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("animelens")

TRACE_MOE_URL = "https://api.trace.moe/search"
SAUCENAO_URL = "https://saucenao.com/search.php"
ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
SAUCENAO_API_KEY = os.getenv("SAUCENAO_API_KEY", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", "").strip()
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL", "https://api.openai.com"
).strip().rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini").strip()
TRACE_CONFIDENCE_THRESHOLD = 0.85
ANILIST_TITLE_CACHE: dict[int, str] = {}
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
