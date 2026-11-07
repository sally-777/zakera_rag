import re
from datetime import date, datetime

_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْـ]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# عبارات معناها "مفيش معلومة" في ملف الداتا
NULL_PHRASES = [
    "غير مذكور في المصدر", "غير مذكورة في المصدر", "غير مذكور في المصادر",
    "غير معروف", "غير معروفة", "غير محدد في المصادر", "غير محدد في المصدر", "غير محدد",
    "لا ينطبق", "لا توجد معلومات", "none", "nan", "-", "—",
]
_SEP = r"[\s/:\-–—.،؛,]"
_NULL_RE = re.compile(
    r"^\s*(?:%s)(?:%s+(?P<rest>.*))?$" % ("|".join(re.escape(p) for p in sorted(NULL_PHRASES, key=len, reverse=True)), _SEP),
    re.S | re.I,
)


def normalize_ar(s) -> str:
    """توحيد الحروف العربية للمقارنة والبحث (مش للعرض)."""
    s = str(s or "").translate(_AR_DIGITS)
    s = _TASHKEEL.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = re.sub(r"[«»\"“”'‘’()\[\]{}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def key_tokens(s) -> set:
    """كلمات مفتاحية: من غير 'ال' وكلمات عامة، عشان نطابق اسم الحرفة مع السؤال."""
    stop = {"صناعه", "حرفه", "فن", "في", "من", "على", "عن", "ايه", "ازاي", "اي", "هي", "هو", "يعني", "بتاع", "بتاعه"}
    out = set()
    for w in re.findall(r"[\w]+", normalize_ar(s)):
        if w.startswith("ال") and len(w) > 4:
            w = w[2:]
        if len(w) >= 3 and w not in stop:
            out.add(w)
    return out


def cell_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def clean_answer(v) -> str:
    """ينضف إجابة خلية. يرجّع '' لو مفيش معلومة فعلية.
    'غير معروف. تذكر المصادر أن...' -> 'تذكر المصادر أن...' (بنشيل العبارة الفاضية ونحتفظ بالباقي)."""
    t = cell_text(v)
    if not t:
        return ""
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    for _ in range(3):  # ممكن تتكرر: "غير معروف/غير مذكور في المصدر/..."
        m = _NULL_RE.match(t)
        if not m:
            break
        t = (m.group("rest") or "").strip()
        if not t:
            return ""
    return t
