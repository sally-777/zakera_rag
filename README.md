# ذاكرة — شات بوت RAG (مصر في ذاكرة المستقبل)

## التشغيل (5 دقايق، من غير GPU ومن غير فلوس)
```bash
python -m venv venv && source venv/bin/activate        # ويندوز: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # ويندوز: copy .env.example .env
# حط ملف الإكسيل في data/Egypt_Craft_Documentation.xlsx
python ingest.py                                        # فهرسة + تقرير جودة الداتا
python chat_cli.py                                      # جرّب من الترمنال
uvicorn app:app --reload --port 8000                    # شغّل الـ API  (توثيق تفاعلي: /docs)
```
أول مرة `fastembed` هينزّل موديل الـ embeddings (~2GB لـ e5-large). لو بطيء: `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
و `MAX_CHUNK_CHARS=450` (الموديل ده بيقطع النص بعد ~128 توكن). ولو مش عايزين embeddings خالص: `EMBEDDER=tfidf`.

## LLM مجاني (اختار واحد في .env)
| المزود | إزاي |
|---|---|
| Gemini (free tier) | مفتاح من Google AI Studio -> `GEMINI_API_KEY=` |
| Groq (free tier) | مفتاح من console.groq.com -> `GROQ_API_KEY=` |
| Ollama (محلي) | `ollama pull qwen2.5:3b` (محتاج ~8GB RAM) |
| من غير LLM | الشات بوت يعرض المقاطع المسترجعة مباشرة (مناسب لتجربة الفرونت) |

أسماء الموديلات المجانية بتتغير؛ لو موديل اتقفل غيّره من `GEMINI_MODEL` / `GROQ_MODEL`. ولو الـ LLM وقع أو خلصت الكوتا، النظام بيرجع تلقائيًا للعرض المباشر.

## عقد الـ API (ده اللي الفرونت/الباك بيشتغلوا عليه)
`POST /chat` ← `{"question": "ازاي بيعملوا المشبك الدمياطي؟", "session_id": "user-1"}`
→ `{"answer": "...[1]", "sources": [{"id","title","publisher","date","url"}], "references": [{"n","craft","governorate","section","record_id","score"}], "found": true, "mode": "gemini"}`

`GET /health` · `GET /crafts` (قائمة الحرف للاقتراحات) · `POST /reindex` (هيدر `X-Admin-Token` لو `ADMIN_TOKEN` متحدد)
الأفاتار بعدين يكلّم نفس `/chat` ويأخد `answer` يحوّله لصوت.

## لما فريق الداتا يضيف بيانات
- الملف الأصلي **مش بيتعدّل**. التنضيف بيتم في الذاكرة وقت الفهرسة.
- `WATCH_DATA=1` (الافتراضي): السيرفر يلاحظ تعديل الملف ويعيد الفهرسة لوحده (الفهرس القديم يفضل شغال لو حصل خطأ).
- أو يدويًا: `python ingest.py` أو `POST /reindex`. الـ embeddings بتتحسب للقطع الجديدة/المعدلة بس.
- لو شكل الملف اتغير (اسم عمود/شيت) الفهرسة بتقف برسالة واضحة. أسماء الشيتات في `config.py`.
- **`index/quality_report.md`** بعد كل فهرسة: مشاكل الداتا + نسبة امتلاء كل سؤال، تبعتوه لفريق الداتا.

## ماذا ينضّف الكود
- يشيل «غير مذكور في المصدر / غير معروف / لا ينطبق» ويحتفظ بالباقي (`غير معروف. تذكر المصادر أن…` ← `تذكر المصادر أن…`).
- يستبعد البيانات الشخصية (س1 الاسم، س2 العمر، س40 التواصل، س39) — `EXCLUDE_QUESTIONS`.
- يستبعد سجلات المثال (C000001/R000001) ويفلتر بحالة المراجعة — `ALLOWED_STATUS` (حاليًا `*` لأن كل السجلات «مسودة»؛ للنشر الفعلي حطوا الحالة المعتمدة).
- يربط كل قطعة بمصادرها من شيت «الأدلة» و«المصادر» عشان الإجابة تطلع بمراجع.
- يتجاهل الأعمدة/الصفوف الفاضية ويقرأ بأسماء الأعمدة مش ترتيبها.

## هيكل المشروع
`config.py` الإعدادات · `ingest.py` الفهرسة + تقرير الجودة · `app.py` الـ API · `chat_cli.py` تجربة سريعة
`rag/loader.py` قراءة الإكسيل · `rag/chunker.py` التقطيع · `rag/index.py` البحث الهجين · `rag/llm.py` المزودين · `rag/pipeline.py` خط الـ RAG
