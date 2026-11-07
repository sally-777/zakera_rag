"""تحويل السجلات المنضّفة لقطع (chunks) صغيرة ومتجمعة حسب الموضوع، وكل قطعة معاها هيدر
فيه اسم الحرفة والمحافظة عشان الـ retrieval يلاقيها حتى لو السؤال ما ذكرش الاسم في نص الإجابة."""
import hashlib
import re

import config
from .textutil import key_tokens

GROUPS = [
    ("تعريف الحرفة ومنتجاتها", [6, 7, 8]),
    ("المكان والممارسة", [3, 4, 5, 9]),
    ("التاريخ والمناسبات", [10, 11]),
    ("الخامات والأدوات", [12, 13]),
    ("خطوات الصناعة", [14, 15, 16, 17]),
    ("وضع الحرفة وعدد الممارسين", [18, 19, 20, 21, 22, 30]),
    ("المهارات والتعلم", [23, 24, 25, 26]),
    ("التحديات والدعم والطلب", [27, 28, 29, 31, 32, 33]),
    ("البيع والتسويق", [34, 35, 36, 37, 38]),
]
OTHER = "معلومات أخرى"


def _looks_like_person(name, enterers):
    return any(name and name == e for e in enterers)


def craft_name_for(rec, crafts, enterers):
    c = crafts.get(rec["craft_id"], {})
    name = c.get("name", "")
    if not name or _looks_like_person(name, enterers):
        # احتياطي: أول جملة من س6 (اسم الحرفة وأسماؤها المحلية)
        s6 = rec["answers"].get(6, "")
        name = re.split(r"[،.\n:؛]", s6.replace("اسم الحرفة:", "").strip())[0][:50].strip() or rec["craft_id"]
    return name


def _split_long(text, limit):
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[\n.؟!])\s*", text):
        if len(cur) + len(piece) + 1 > limit and cur:
            parts.append(cur.strip())
            cur = ""
        cur += piece + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _sources_for(rec, qs, data):
    ev = data["evidence"].get(rec["record_id"], {})
    ids = []
    for q in qs:
        for s in ev.get(q, []):
            if s not in ids:
                ids.append(s)
    if not ids:  # احتياطي: كل مصادر السجل
        for q in sorted(ev):
            for s in ev[q]:
                if s not in ids:
                    ids.append(s)
    return [data["sources"][s] for s in ids if s in data["sources"]]


def build_chunks(data) -> list:
    crafts, labels = data["crafts"], data["question_labels"]
    enterers = {r.get("enterer") for r in data["records"] if r.get("enterer")}
    chunks, stats = [], {"skipped_status": 0, "skipped_example": 0}
    allowed = config.ALLOWED_STATUS

    # --- قطع تعريف الحرف من «دليل الحرف» ---
    for cid, c in crafts.items():
        if cid in config.EXAMPLE_IDS or not c["description"] or _looks_like_person(c["name"], enterers):
            continue
        text = f"الحرفة: {c['name']}" + (f" (أسماء أخرى: {c['alt_names']})" if c["alt_names"] else "") + f"\n{c['description']}"
        chunks.append({"id": f"{cid}-guide", "text": text, "craft_id": cid, "craft_name": c["name"],
                       "record_id": None, "governorate": "", "section": "تعريف عام", "questions": [],
                       "sources": [], "alt_names": c["alt_names"]})

    # --- قطع السجلات ---
    for rec in data["records"]:
        if rec["record_id"] in config.EXAMPLE_IDS or rec["craft_id"] in config.EXAMPLE_IDS:
            stats["skipped_example"] += 1
            continue
        if "*" not in allowed and rec.get("review_status", "") not in allowed:
            stats["skipped_status"] += 1
            continue
        cname = craft_name_for(rec, crafts, enterers)
        place = "، ".join(x for x in [rec.get("center"), rec.get("village")] if x)
        header = f"الحرفة: {cname} | المحافظة: {rec.get('governorate') or 'غير محددة'}" + (f" | المنطقة: {place}" if place else "")
        if rec.get("period"):
            header += f" | الفترة التي تصفها المعلومات: {rec['period'][:80]}"

        grouped, used = [], set()
        for title, qs in GROUPS:
            present = [q for q in qs if q in rec["answers"] and q not in config.EXCLUDE_QUESTIONS]
            if present:
                grouped.append((title, present))
                used.update(present)
        rest = [q for q in sorted(rec["answers"]) if q not in used and q not in config.EXCLUDE_QUESTIONS]
        if rest:
            grouped.append((OTHER, rest))

        for gi, (title, qs) in enumerate(grouped):
            items = []
            for q in qs:
                label = labels.get(q, f"سؤال {q}")
                for j, piece in enumerate(_split_long(rec["answers"][q], config.MAX_CHUNK_CHARS)):
                    items.append((q, f"- {label}: {piece}" if j == 0 else f"- {label} (تكملة): {piece}"))
            # تجميع العناصر في قطع بحد أقصى للطول
            part, cur, cur_q = 0, [], []
            for q, line in items + [(None, None)]:
                if line is None or (cur and sum(len(x) for x in cur) + len(line) > config.MAX_CHUNK_CHARS):
                    if cur:
                        body = "\n".join(cur)
                        text = f"{header}\n[{title}]\n{body}"
                        chunks.append({
                            "id": f"{rec['record_id']}-{gi}-{part}", "text": text,
                            "craft_id": rec["craft_id"], "craft_name": cname, "record_id": rec["record_id"],
                            "governorate": rec.get("governorate", ""), "section": title,
                            "questions": sorted(set(cur_q)), "sources": _sources_for(rec, cur_q, data),
                            "alt_names": crafts.get(rec["craft_id"], {}).get("alt_names", ""),
                        })
                        part += 1
                    cur, cur_q = [], []
                if line is not None:
                    cur.append(line)
                    cur_q.append(q)

    for c in chunks:
        c["hash"] = hashlib.sha1(c["text"].encode("utf-8")).hexdigest()
        c["craft_keys"] = sorted(key_tokens(c["craft_name"]))
    return chunks, stats
