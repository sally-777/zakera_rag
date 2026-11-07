"""الفهرس: بحث هجين = (embeddings دلالية لو متاحة) + (TF-IDF بحروف وكلمات). كله بيشتغل على CPU.
الـ embeddings بتتخزن بالـ hash بتاع كل قطعة، فإعادة الفهرسة بتحسب بس القطع الجديدة/المعدلة."""
import json
import logging
import pickle
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from .textutil import key_tokens, normalize_ar

log = logging.getLogger("zakera.index")


# كلمات في السؤال -> القسم المتوقع (بتحسّن اختيار القطعة الصح داخل الحرفة)
SECTION_HINTS = {
    "الخامات والأدوات": ["خام", "مواد", "ماده", "ادوات", "اداه", "معدات", "بتتصنع من", "بيتعمل من", "مصنوع"],
    "خطوات الصناعة": ["ازاي بيعمل", "ازاي بتتعمل", "ازاي بيصنع", "ازاي بتتصنع", "ازاي يتعمل", "طريقه", "خطوات", "مراحل", "تصنيع", "بتتعمل", "بيتعمل", "ياخد وقت", "مده"],
    "التحديات والدعم والطلب": ["مشاكل", "مشكله", "تحديات", "صعوبات", "دعم", "طلب", "تكلفه", "غلا", "اسعار", "خطر", "تواجه", "عقبات"],
    "وضع الحرفة وعدد الممارسين": ["عدد", "كام حرفي", "ممارسين", "حرفيين", "شباب", "اندثار", "تندثر", "انقراض", "بتقل", "هتختفي", "مستقبل"],
    "المهارات والتعلم": ["مهارات", "تتعلم", "اتعلم", "تعلم", "تدريب", "تتدرب", "موهبه"],
    "التاريخ والمناسبات": ["تاريخ", "اصل", "مناسبات", "مواسم", "عادات", "رمضان", "فرح", "قديم", "ظهرت", "بدات"],
    "البيع والتسويق": ["بيع", "تسويق", "سعر", "بكام", "اشتري", "بيتباع", "شحن", "توصيل", "معارض"],
    "تعريف الحرفة ومنتجاتها": ["ايه هي", "يعني ايه", "تعريف", "منتجات", "بتنتج", "وصف", "قولي عن", "احكي عن", "عن حرفه", "نبذه"],
    "المكان والممارسة": ["فين", "مكان", "بتتمارس", "اماكن", "قريه", "منطقه"],
}


class DenseEmbedder:
    def __init__(self, model_name):
        from fastembed import TextEmbedding  # import متأخر: اختياري
        self.name = model_name
        self.model = TextEmbedding(model_name=model_name)

    @staticmethod
    def _norm(a):
        a = np.asarray(a, dtype="float32")
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    def passages(self, texts):
        return self._norm(list(self.model.passage_embed(texts, batch_size=16)))

    def query(self, q):
        return self._norm(list(self.model.query_embed([q])))[0]


def get_dense():
    if config.EMBEDDER == "tfidf":
        return None
    try:
        return DenseEmbedder(config.EMBED_MODEL)
    except Exception as e:  # مش متركب أو مفيش نت لتحميل الموديل
        if config.EMBEDDER == "fastembed":
            raise
        log.warning("الـ embeddings الدلالية مش متاحة (%s). هكمل بالبحث النصي TF-IDF بس.", e)
        return None


def _vectorizers():
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=normalize_ar)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, preprocessor=normalize_ar,
                           token_pattern=r"(?u)\b\w\w+\b")
    return char, word


class Index:
    def __init__(self, chunks, char, word, Xc, Xw, dense=None, dense_model=None, dense_cache=None):
        self.chunks, self.char, self.word, self.Xc, self.Xw = chunks, char, word, Xc, Xw
        self.dense, self.dense_model = dense, dense_model
        self.dense_cache = dense_cache or {}
        self._embedder = None
        self.gov_keys = {c["governorate"]: key_tokens(c["governorate"]) for c in chunks if c["governorate"]}
        # مفاتيح كل حرفة: الاسم المعتمد + كل اسم بديل
        self.craft_sets, df = {}, {}
        for c in chunks:
            if c["craft_id"] in self.craft_sets:
                continue
            sets = [set(c["craft_keys"])]
            for alt in re.split(r"[،,/؛]|\sأو\s", c.get("alt_names") or ""):
                ks = key_tokens(alt)
                if ks:
                    sets.append(ks)
            self.craft_sets[c["craft_id"]] = [x for x in sets if x]
            for t in set(c["craft_keys"]):
                df[t] = df.get(t, 0) + 1
        self.gov_tokens = set().union(*self.gov_keys.values()) if self.gov_keys else set()
        for cid, sets in self.craft_sets.items():  # «دمياط» في اسم حرفة ما تتعدش تطابق حرفة
            self.craft_sets[cid] = [x - self.gov_tokens for x in sets if x - self.gov_tokens]
        self.token_df = {t: n for t, n in df.items() if t not in self.gov_tokens}

    # ---------- بناء ----------
    @classmethod
    def build(cls, chunks, previous=None):
        texts = [c["text"] for c in chunks]
        char, word = _vectorizers()
        Xc, Xw = char.fit_transform(texts), word.fit_transform(texts)
        embedder = get_dense()
        dense, cache = None, {}
        if embedder is not None:
            if previous and previous.dense_model == embedder.name:
                cache = dict(previous.dense_cache)
            todo = [i for i, c in enumerate(chunks) if c["hash"] not in cache]
            log.info("embeddings: %d جديدة/معدلة، %d من الكاش", len(todo), len(chunks) - len(todo))
            if todo:
                vecs = embedder.passages([texts[i] for i in todo])
                for i, v in zip(todo, vecs):
                    cache[chunks[i]["hash"]] = v
            live = {c["hash"] for c in chunks}
            cache = {h: v for h, v in cache.items() if h in live}
            dense = np.stack([cache[c["hash"]] for c in chunks])
        idx = cls(chunks, char, word, Xc, Xw, dense, embedder.name if embedder else None, cache)
        idx._embedder = embedder
        return idx

    # ---------- حفظ وتحميل ----------
    def save(self, directory=None):
        d = directory or config.INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "index.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "char": self.char, "word": self.word, "Xc": self.Xc, "Xw": self.Xw,
                         "dense": self.dense, "dense_model": self.dense_model, "dense_cache": self.dense_cache}, f)
        (d / "chunks.json").write_text(json.dumps(
            [{k: v for k, v in c.items() if k != "craft_keys"} for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8")

    @classmethod
    def load(cls, directory=None):
        p = (directory or config.INDEX_DIR) / "index.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            o = pickle.load(f)
        return cls(o["chunks"], o["char"], o["word"], o["Xc"], o["Xw"], o["dense"], o["dense_model"], o["dense_cache"])

    # ---------- بحث ----------
    def _query_embedder(self):
        if self.dense is None:
            return None
        if self._embedder is None:
            try:
                self._embedder = DenseEmbedder(self.dense_model)
            except Exception as e:
                log.warning("مش قادر أحمّل موديل الـ embeddings وقت البحث (%s)، هستخدم TF-IDF بس", e)
                self.dense = None
                return None
        return self._embedder

    def _craft_match(self, qk):
        """أي حرفات السؤال ذكرها؟ بنطابق الاسم المعتمد + الأسماء البديلة،
        وكمان كلمة مميزة (بتظهر في اسم حرفتين بالكتير) تكفي — عشان «الدباغة» تجيب «الدباغة التقليدية للجلود»."""
        hit = {}
        for cid, sets in self.craft_sets.items():
            best = 0.0
            for ks in sets:
                common = ks & qk
                if not common:
                    continue
                frac = len(common) / len(ks)
                if frac >= 0.5:
                    best = max(best, frac)
                elif any(len(t) >= 4 and self.token_df.get(t, 99) <= 2 for t in common):
                    best = max(best, 0.6)
            if best:
                hit[cid] = best
        return hit

    def search(self, query, k=None, min_score=None):
        k = k or config.TOP_K
        q = normalize_ar(query)
        qc, qw = self.char.transform([q]), self.word.transform([q])
        lex = 0.5 * (self.Xc @ qc.T).toarray().ravel() + 0.5 * (self.Xw @ qw.T).toarray().ravel()
        emb = self._query_embedder()
        score = (0.6 * (self.dense @ emb.query(query)) + 0.4 * lex) if emb is not None else lex.copy()

        qk = key_tokens(query)
        craft_hit = self._craft_match(qk - self.gov_tokens)
        gov_hit = {g for g, gk in self.gov_keys.items() if gk and gk <= qk}
        # نية السؤال -> القسم (الكلمة لازم تبدأ عند حدّ كلمة، مع السماح بـ ال/و/ب/ل قبلها)
        sections = {sec for sec, kws in SECTION_HINTS.items()
                    if any(re.search(r"(?<!\w)(?:ال|و|ب|ل|ف)?" + re.escape(kw), q) for kw in kws)}

        for i, c in enumerate(self.chunks):
            if c["craft_id"] in craft_hit:
                score[i] += 0.25 * craft_hit[c["craft_id"]]
            if c["governorate"] in gov_hit:
                score[i] += 0.08
            if c["section"] in sections:
                score[i] += 0.12
            if c["record_id"] is None and "تعريف الحرفة ومنتجاتها" in sections:
                score[i] += 0.12
        order = np.argsort(-score)[:k]
        ms = config.MIN_SCORE if min_score is None else min_score
        hits = [(self.chunks[i], float(score[i])) for i in order if score[i] >= ms]
        signals = {"crafts": set(craft_hit), "governorates": gov_hit, "sections": sections,
                   "list_intent": ("حرف" in qk) and not craft_hit}
        return hits, signals

    def crafts_overview(self, governorate=None, limit=12):
        """قطعة تعريف واحدة لكل حرفة (اختياريًا في محافظة معينة) — لأسئلة «الحرف اللي في…»."""
        seen, out = set(), []
        for c in self.chunks:
            if c["section"] != "تعريف الحرفة ومنتجاتها" or c["craft_id"] in seen:
                continue
            if governorate and c["governorate"] != governorate:
                continue
            seen.add(c["craft_id"])
            out.append(c)
        return out[:limit]
