"""كل الإعدادات من مكان واحد. بتتقرأ من متغيرات البيئة أو ملف .env"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _load_dotenv():
    p = BASE / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else BASE / p


DATA_FILE = _path(os.getenv("DATA_FILE", "data/Egypt_Craft_Documentation.xlsx"))
INDEX_DIR = _path(os.getenv("INDEX_DIR", "index"))

ALLOWED_STATUS = [s.strip() for s in os.getenv("ALLOWED_STATUS", "*").split(",") if s.strip()]
EXCLUDE_QUESTIONS = {int(x) for x in os.getenv("EXCLUDE_QUESTIONS", "1,2,39,40").split(",") if x.strip()}
WATCH_DATA = os.getenv("WATCH_DATA", "1") == "1"

EMBEDDER = os.getenv("EMBEDDER", "auto")  # auto | fastembed | tfidf
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto | gemini | groq | ollama | none
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.08"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1400"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# أسماء الشيتات في ملف الإكسيل (لو فريق الداتا غيّرها، عدّلها هنا بس)
SHEET_RECORDS = "سجلات التوثيق"
SHEET_CRAFTS = "دليل الحرف"
SHEET_SOURCES = "المصادر"
SHEET_EVIDENCE = "الأدلة"
SHEET_REVIEW = "المراجعة"

# سجلات المثال اللي في التعليمات (مش بيانات حقيقية)
EXAMPLE_IDS = {"C000001", "R000001"}
