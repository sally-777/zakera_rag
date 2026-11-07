"""طبقة الـ LLM. كلها بتتكلم HTTP عادي (requests) من غير مكتبات إضافية.
المزودين المجانيين: Gemini (free tier) / Groq (free tier) / Ollama (محلي) / none (من غير LLM)."""
import logging
import os
import time
 
import requests
 
import config
 
log = logging.getLogger("zakera.llm")
 
 
class LLMError(Exception):
    pass
 
 
RETRY_STATUS = {429, 500, 502, 503, 504}  # أخطاء مؤقتة (ضغط على الموديل / حد الاستخدام)
 
 
def _post(url, headers, payload, timeout=60, retries=3):
    """يعيد المحاولة تلقائيًا (2ث ثم 4ث ثم 8ث) لو الخطأ مؤقت زي 503 و429."""
    delay = 2
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            if attempt == retries:
                raise LLMError(f"مشكلة اتصال: {e}")
            time.sleep(delay); delay *= 2
            continue
        if r.status_code in RETRY_STATUS and attempt < retries:
            log.info("خطأ مؤقت %s — محاولة تانية بعد %sث", r.status_code, delay)
            time.sleep(delay); delay *= 2
            continue
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code}: {r.text[:300]}")
        return r.json()
 
 
class Gemini:
    name = "gemini"
 
    def _models(self):
        extra = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-2.0-flash").split(",") if m.strip()]
        out = []
        for m in [config.GEMINI_MODEL] + extra:
            if m not in out:
                out.append(m)
        return out
 
    def generate(self, system, messages, temperature=0.2):
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                "generationConfig": {"temperature": temperature}}
        last = None
        for model in self._models():  # لو موديل مشغول أو اتقفل نجرّب اللي بعده
            try:
                data = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                             {"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
                             body, retries=2)
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
                if text:
                    log.info("رد من %s", model)
                    return text
                last = LLMError(f"رد فاضي من {model}")
            except (LLMError, KeyError, IndexError) as e:
                log.warning("الموديل %s فشل: %s", model, str(e)[:200])
                last = e
        raise LLMError(f"كل موديلات Gemini فشلت. آخر خطأ: {last}")
 
 
class Groq:
    name = "groq"
 
    def generate(self, system, messages, temperature=0.2):
        data = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            {"model": config.GROQ_MODEL, "temperature": temperature,
             "messages": [{"role": "system", "content": system}] + messages})
        return data["choices"][0]["message"]["content"].strip()
 
 
class Compat:
    """أي مزود بنظام OpenAI (OpenRouter / Mistral / Cerebras / Together ...). كله من .env:
    LLM_BASE_URL و LLM_API_KEY و LLM_MODEL"""
    name = "compat"
 
    def generate(self, system, messages, temperature=0.2):
        base = os.getenv("LLM_BASE_URL", "").rstrip("/")
        data = _post(
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"},
            {"model": os.getenv("LLM_MODEL", ""), "temperature": temperature,
             "messages": [{"role": "system", "content": system}] + messages})
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise LLMError(f"رد غير متوقع: {str(data)[:300]}")
 
 
class Ollama:
    name = "ollama"
 
    def generate(self, system, messages, temperature=0.2):
        data = _post(
            f"{config.OLLAMA_URL}/api/chat", {},
            {"model": config.OLLAMA_MODEL, "stream": False, "options": {"temperature": temperature},
             "messages": [{"role": "system", "content": system}] + messages}, timeout=180)
        return data["message"]["content"].strip()
 
 
def _ollama_up():
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=1.5).ok
    except Exception:
        return False
 
 
class Chain:
    """يجرّب المزودين بالترتيب: لو الأول فشل ينتقل للتاني."""
 
    def __init__(self, providers):
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
 
    def generate(self, system, messages, temperature=0.2):
        last = None
        for p in self.providers:
            try:
                return p.generate(system, messages, temperature)
            except Exception as e:
                log.warning("المزود %s فشل: %s", p.name, str(e)[:200])
                last = e
        raise LLMError(str(last))
 
 
def get_llm():
    """يرجّع LLM أو None (= وضع الاستخراج من غير LLM). في auto بنرتب: gemini ثم groq ثم ollama."""
    p = config.LLM_PROVIDER
    if p == "none":
        return None
    chain = []
    if p == "gemini" or (p == "auto" and config.GEMINI_API_KEY):
        chain.append(Gemini())
    if p == "groq" or (p == "auto" and config.GROQ_API_KEY):
        chain.append(Groq())
    if p == "compat" or (p == "auto" and os.getenv("LLM_API_KEY")):
        chain.append(Compat())
    if p == "ollama" or (p == "auto" and _ollama_up()):
        chain.append(Ollama())
    if not chain:
        return None
    return chain[0] if len(chain) == 1 else Chain(chain)
 
