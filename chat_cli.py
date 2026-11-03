"""جرّب الشات بوت من الترمنال من غير API:  python chat_cli.py"""
import logging
from rag.pipeline import RagEngine

logging.basicConfig(level=logging.WARNING)
e = RagEngine()
print("اكتب سؤالك (exit للخروج). حالة النظام:", e.info())
while True:
    q = input("\nأنت: ").strip()
    if q in ("exit", "خروج", ""):
        break
    r = e.ask(q, "cli")
    print("\n" + r["answer"])
    for s in r["sources"][:5]:
        print(f"   [{s['n']}] {s['title']} — {s['publisher']} {s['url']}")