"""API الشات بوت. الفرونت/الباك بيتعاملوا مع العقد ده بس:
   POST /chat    {question, session_id?}  ->  {answer, sources[], references[], found}
   GET  /health  ,  GET /crafts  ,  POST /reindex
شغّله:  uvicorn app:app --reload --port 8000     (والتوثيق التفاعلي على /docs)"""
import logging
import os
import threading
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import ingest
from rag import loader
from rag.pipeline import RagEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("zakera.api")

app = FastAPI(title="Zakera RAG API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # للديمو

engine: RagEngine | None = None
_reindex_lock = threading.Lock()
_state = {"last_indexed": None, "last_error": None, "data_mtime": None}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str = "default"


def reindex() -> dict:
    """يعيد الفهرسة من ملف الإكسيل ويحمّل الفهرس الجديد في الذاكرة (من غير إيقاف السيرفر)."""
    global engine
    with _reindex_lock:
        try:
            rep = ingest.run()
            if engine is None:
                engine = RagEngine()
            else:
                engine.reload()
            _state.update(last_indexed=time.strftime("%Y-%m-%d %H:%M:%S"), last_error=None,
                          data_mtime=config.DATA_FILE.stat().st_mtime)
            return {"ok": True, "records": rep["records"], "chunks": rep["chunks"], "issues": len(rep["issues"])}
        except (loader.SchemaError, Exception) as e:
            _state["last_error"] = str(e)  # الفهرس القديم يفضل شغال
            log.error("فشل إعادة الفهرسة: %s", e)
            return {"ok": False, "error": str(e)}


def _watcher():
    while True:
        time.sleep(30)
        try:
            m = config.DATA_FILE.stat().st_mtime
            if _state["data_mtime"] and m != _state["data_mtime"]:
                log.info("الداتا اتعدلت — بعيد الفهرسة...")
                time.sleep(3)  # نستنى الملف يخلص حفظ
                reindex()
        except FileNotFoundError:
            pass


@app.on_event("startup")
def startup():
    global engine
    if (config.INDEX_DIR / "index.pkl").exists():
        engine = RagEngine()
        _state["data_mtime"] = config.DATA_FILE.stat().st_mtime if config.DATA_FILE.exists() else None
    elif config.DATA_FILE.exists():
        reindex()
    else:
        log.warning("ملف الداتا مش موجود: %s", config.DATA_FILE)
    if config.WATCH_DATA:
        threading.Thread(target=_watcher, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok" if engine and engine.ready else "no_index", **(engine.info() if engine else {}), **_state}


@app.post("/chat")
def chat(req: ChatRequest):
    if not engine or not engine.ready:
        raise HTTPException(503, "الفهرس لسه مش جاهز")
    return engine.ask(req.question, req.session_id)


@app.get("/crafts")
def crafts():
    """قائمة الحرف المفهرسة (للفرونت: اقتراحات أسئلة / قائمة اختيار)."""
    if not engine or not engine.ready:
        raise HTTPException(503, "الفهرس لسه مش جاهز")
    seen = {}
    for c in engine.index.chunks:
        seen.setdefault(c["craft_id"], {"craft_id": c["craft_id"], "name": c["craft_name"], "governorates": set()})
        if c["governorate"]:
            seen[c["craft_id"]]["governorates"].add(c["governorate"])
    return [{**v, "governorates": sorted(v["governorates"])} for v in seen.values()]


@app.post("/reindex")
def reindex_endpoint(x_admin_token: str = Header(default="")):
    if config.ADMIN_TOKEN and x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(401, "غير مصرح")
    return reindex()
