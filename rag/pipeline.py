"""خط الـ RAG كامل: سؤال -> بحث -> بناء السياق -> LLM -> إجابة + مصادر."""
import logging
import re
import threading
from collections import defaultdict, deque

import config
from .index import Index
from .llm import LLMError, get_llm
from .textutil import normalize_ar

log = logging.getLogger("zakera.rag")

SYSTEM_PROMPT = """أنت «ذاكرة»، المساعد الذكي لمبادرة «مصر في ذاكرة المستقبل» التابعة لوزارة الثقافة، تجيب الجمهور عن الحرف التقليدية المصرية اعتمادًا على قاعدة بيانات موثقة.

قواعد المحتوى:
1. أجب من «السياق» المرفق فقط. يُمنع إضافة أي معلومة من عندك أو من معرفتك العامة حتى لو كنت تعرفها.
2. إن لم يتضمن السياق إجابة السؤال، فقل بلطف ووضوح إن هذه المعلومة غير موثقة حاليًا في قاعدة بيانات «ذاكرة»، واقترح سؤالًا قريبًا يمكنك الإجابة عنه.
3. اذكر رقم المصدر بين أقواس مربعة بعد كل معلومة، مثل [1] أو [2][3]، باستخدام الأرقام المكتوبة في خانة «المصدر:» فوق كل مقطع لا ترتيب المقطع. لا تكتب قائمة مصادر في نهاية الرد، فالنظام يعرضها وحده.
4. إذا اختلفت المصادر في معلومة فوضّح الاختلاف، وإذا جاءت المعلومة من مصدر واحد فأشر إلى ذلك.
5. لا تذكر أي بيانات شخصية عن الحرفيين، مثل أرقام الهواتف أو العناوين الخاصة.
6. إذا كان السؤال لا علاقة له بالحرف والتراث المصري الموثق، فاعتذر بلطف وبيّن أن تخصصك هو الحرف الموثقة في «ذاكرة».

قواعد الأسلوب:
7. اكتب بفصحى مبسطة سليمة، بنبرة محترمة ودافئة تناسب جميع الأعمار والمستويات، وبجُمل قصيرة واضحة. وإن جاء السؤال بالعامية المصرية فأجب بالفصحى المبسطة نفسها. ضع المصطلحات الحرفية المحلية بين «» مع شرح موجز عند أول ذكر لها.
8. اكتب نصًا عاديًا بلا تنسيق Markdown: لا نجوم ولا عناوين ولا جداول.
9. التزم ببنية ثابتة في كل رد:
   - جملة تمهيدية قصيرة تجيب مباشرة عن السؤال.
   - ثم التفاصيل: خطوات مرقمة (1. 2. 3.) إن كان السؤال عن طريقة الصنع، أو نقاط قصيرة تبدأ بشرطة (-) في غير ذلك، وكل نقطة أو خطوة في سطر مستقل، وبحد أقصى ثماني نقاط.
   - ثم، عند الحاجة فقط، سطر ختامي واحد يبدأ بكلمة «ملاحظة:».
10. لا تكرر نص السؤال، ولا تبدأ بعبارات مثل «بالتأكيد» أو «بناءً على السياق»."""

NOT_FOUND = ("هذه المعلومة غير موثقة حتى الآن في قاعدة بيانات «ذاكرة». "
             "يمكنك السؤال عن حرفة بعينها، مثل المشبك الدمياطي أو الفخار، أو عن خطواتها وخاماتها والتحديات التي تواجهها.")


def _number_sources(hits):
    """ترقيم المصادر الفريدة بالترتيب: {id: رقم}. نفس الأرقام بتظهر في الرد وفي قائمة sources."""
    order = {}
    for c, _ in hits:
        for s in c["sources"]:
            if s["id"] not in order:
                order[s["id"]] = len(order) + 1
    return order


def _format_context(hits, src_no):
    blocks = []
    for i, (c, _) in enumerate(hits, 1):
        nums = "".join(f"[{src_no[s['id']]}]" for s in c["sources"]) or "غير محدد"
        blocks.append(f"--- مقطع {i} | المصدر: {nums} ---\n{c['text']}")
    return "\n\n".join(blocks)


def _extractive_answer(hits):
    """وضع من غير LLM: بيعرض أحسن المقاطع مباشرة."""
    lines = ["أقرب ما وُجد في قاعدة البيانات:\n"]
    for n, (c, _) in enumerate(hits[:3], 1):
        body = c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]
        lines.append(f"[{n}] {c['craft_name']} — {c['section']}\n{body}\n")
    return "\n".join(lines)


_FOLLOWUP_RE = re.compile(r"(?<!\w)(طب|كمان|دي|ده|دول)(?!\w)|\w{3,}(ها|هم|هما)(?!\w)")


def _is_followup(question, sig):
    if sig["crafts"] or sig["governorates"] or sig["list_intent"] or len(question.split()) > 6:
        return False
    qn = normalize_ar(question)
    return qn.startswith("و") or bool(_FOLLOWUP_RE.search(qn))


def clean_answer(text):
    """تنضيف رد الموديل: يشيل علامات Markdown ويفصل الخطوات المرقمة الملزوقة في بعض."""
    t = re.sub(r"\*\*|__", "", text)                                # **عريض**
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)                  # # عناوين
    t = re.sub(r"^\s*[*•]\s+", "- ", t, flags=re.M)                  # * نقطة -> - نقطة
    t = t.replace("*", "")                                           # أي نجمة متبقية
    t = re.sub(r"(?<=[.\]])(?=\d{1,2}\.\s)", "\n", t)                # "...الزائدة.5. التبريد" -> سطر جديد
    t = re.sub(r"(?<=\S)[ \t]+(?=\d{1,2}\.\s[^\d])", "\n", t) if "\n" not in t else t
    t = re.sub(r"^---+\s*$", "", t, flags=re.M)                      # فواصل
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


class RagEngine:
    def __init__(self):
        self.index = Index.load()
        self.llm = get_llm()
        self.lock = threading.Lock()
        self.history = defaultdict(lambda: deque(maxlen=8))  # آخر 4 تبادلات لكل جلسة
        log.info("RAG جاهز. الفهرس: %s قطعة | LLM: %s", len(self.index.chunks) if self.index else 0,
                 self.llm.name if self.llm else "none (extractive)")

    def reload(self):
        with self.lock:
            self.index = Index.load()

    @property
    def ready(self):
        return self.index is not None

    def info(self):
        return {"ready": self.ready, "chunks": len(self.index.chunks) if self.index else 0,
                "llm": self.llm.name if self.llm else "none",
                "embeddings": (self.index.dense_model if self.index and self.index.dense is not None else "tfidf")}

    def ask(self, question, session_id="default", top_k=None):
        if not self.ready:
            raise RuntimeError("الفهرس لسه مش جاهز. شغّل: python ingest.py")
        question = (question or "").strip()
        hist = self.history[session_id]

        hits, sig = self.index.search(question, k=top_k)

        # سؤال متابعة ("وإيه خاماتها؟"): قصير، مفيهوش اسم حرفة/محافظة، وفيه ضمير أو بيبدأ بـ "و/طب"
        if hist and _is_followup(question, sig):
            last_user = [m["content"] for m in hist if m["role"] == "user"]
            if last_user:
                hits, sig = self.index.search(f"{last_user[-1]} {question}", k=top_k)

        # «الحرف اللي في دمياط؟» -> نعرض حرفة واحدة من كل حرفة في المحافظة
        if sig["list_intent"]:
            gov = next(iter(sig["governorates"]), None)
            over = self.index.crafts_overview(gov, limit=12)
            if over:
                hits = [(dict(c, text=c["text"][:450]), 1.0) for c in over]

        # حارس الأسئلة الخارجة عن الموضوع: لازم السؤال يلمس حرفة/محافظة، أو تطابق نصي قوي.
        # لو فيه LLM وفيه بس "نوع معلومة" (زي: بكام) بنسيب الـ LLM يقرر (القاعدة 8 في البرومبت).
        on_topic = (sig["crafts"] or sig["governorates"] or sig["list_intent"]
                    or (hits and hits[0][1] >= 0.25) or (sig["sections"] and self.llm is not None))
        if not hits or not on_topic:
            return {"answer": NOT_FOUND, "sources": [], "references": [], "found": False}

        src_no = _number_sources(hits)
        answer, mode = None, "extractive"
        if self.llm:
            user_msg = f"السياق:\n{_format_context(hits, src_no)}\n\nالسؤال: {question}"
            msgs = list(hist) + [{"role": "user", "content": user_msg}]
            try:
                answer, mode = clean_answer(self.llm.generate(SYSTEM_PROMPT, msgs)), self.llm.name
            except (LLMError, Exception) as e:  # أي مشكلة في الـ LLM -> نرجع للاستخراج المباشر
                log.warning("فشل الـ LLM (%s) — رجعنا للاستخراج المباشر", e)
        if answer is None:
            answer = _extractive_answer(hits)

        hist.append({"role": "user", "content": question})
        hist.append({"role": "assistant", "content": answer})

        sources, done = [], set()
        for c, _ in hits:
            for s in c["sources"]:
                if s["id"] not in done:
                    done.add(s["id"])
                    sources.append({"n": src_no[s["id"]], **{k: s.get(k, "") for k in ("id", "title", "publisher", "date", "url")}})
        refs = [{"n": n, "craft": c["craft_name"], "governorate": c["governorate"], "section": c["section"],
                 "record_id": c["record_id"], "score": round(sc, 3), "source_ids": [s["id"] for s in c["sources"]]}
                for n, (c, sc) in enumerate(hits, 1)]
        return {"answer": answer, "sources": sources, "references": refs, "found": True, "mode": mode}
 
