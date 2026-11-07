"""قراءة ملف الإكسيل. القراءة بتعتمد على أسماء الأعمدة (مش ترتيبها)،
فلو فريق الداتا أضاف صفوف أو أعاد ترتيب الأعمدة الكود يفضل شغال."""
import re
from collections import defaultdict

import openpyxl

import config
from .textutil import cell_text, clean_answer


class SchemaError(Exception):
    pass


def _rows(wb, name):
    if name not in wb.sheetnames:
        raise SchemaError(f"الشيت «{name}» مش موجود في الملف. الشيتات الموجودة: {wb.sheetnames}")
    return [list(r) for r in wb[name].iter_rows(values_only=True)]


def _header(rows, first_prefix, sheet):
    for i, r in enumerate(rows[:60]):
        if r and r[0] is not None and cell_text(r[0]).startswith(first_prefix) and sum(c is not None for c in r) >= 3:
            return i, [cell_text(c) for c in r]
    raise SchemaError(f"ما لقيتش صف العناوين (اللي بيبدأ بـ «{first_prefix}») في شيت «{sheet}»")


def _col(hdr, prefix, required=True, sheet=""):
    for i, h in enumerate(hdr):
        if h.startswith(prefix):
            return i
    if required:
        raise SchemaError(f"عمود «{prefix}» مش موجود في شيت «{sheet}». لو اتغير اسمه عدّل الكود أو رجّعه.")
    return None


def _get(row, idx):
    return cell_text(row[idx]) if idx is not None and idx < len(row) else ""


def load(path) -> dict:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    warnings = []

    # ---- دليل الحرف ----
    rows = _rows(wb, config.SHEET_CRAFTS)
    h, hdr = _header(rows, "رقم الحرفة", config.SHEET_CRAFTS)
    ci = _col(hdr, "رقم الحرفة", sheet=config.SHEET_CRAFTS)
    cn = _col(hdr, "الاسم المعتمد", sheet=config.SHEET_CRAFTS)
    ca = _col(hdr, "الأسماء المحلية", False)
    cd = _col(hdr, "وصف", False)
    crafts = {}
    for r in rows[h + 1:]:
        cid = _get(r, ci)
        if re.fullmatch(r"C\d{6}", cid):
            crafts[cid] = {
                "id": cid,
                "name": clean_answer(_get(r, cn)),
                "alt_names": clean_answer(_get(r, ca)),
                "description": clean_answer(_get(r, cd)),
            }

    # ---- المصادر ----
    rows = _rows(wb, config.SHEET_SOURCES)
    h, hdr = _header(rows, "رقم المصدر", config.SHEET_SOURCES)
    idx = {k: _col(hdr, p, False) for k, p in [
        ("id", "رقم المصدر"), ("title", "عنوان المصدر"), ("type", "نوع المادة"),
        ("publisher", "الجهة المنتجة"), ("date", "تاريخ النشر"), ("url", "الرابط أو مكان")]}
    sources = {}
    for r in rows[h + 1:]:
        sid = _get(r, idx["id"])
        if re.fullmatch(r"S[EI]\d{6}", sid):
            sources[sid] = {k: clean_answer(_get(r, i)) for k, i in idx.items() if k != "id"}
            sources[sid]["id"] = sid

    # ---- الأدلة: (سجل، سؤال) -> مصادر ----
    rows = _rows(wb, config.SHEET_EVIDENCE)
    h, hdr = _header(rows, "رقم سجل التوثيق", config.SHEET_EVIDENCE)
    er = _col(hdr, "رقم سجل التوثيق", sheet=config.SHEET_EVIDENCE)
    eq = _col(hdr, "رقم السؤال", sheet=config.SHEET_EVIDENCE)
    es = _col(hdr, "رقم المصدر", sheet=config.SHEET_EVIDENCE)
    evidence = defaultdict(lambda: defaultdict(list))
    for r in rows[h + 1:]:
        rid, sid, q = _get(r, er), _get(r, es), _get(r, eq)
        if re.fullmatch(r"R\d{6}", rid) and sid in sources and q:
            try:
                qn = int(float(q))
            except ValueError:
                continue
            if sid not in evidence[rid][qn]:
                evidence[rid][qn].append(sid)

    # ---- المراجعة: حالة كل سجل ----
    rows = _rows(wb, config.SHEET_REVIEW)
    h, hdr = _header(rows, "رقم سجل التوثيق", config.SHEET_REVIEW)
    rr = _col(hdr, "رقم سجل التوثيق", sheet=config.SHEET_REVIEW)
    rs = _col(hdr, "حالة السجل", False)
    ra = _col(hdr, "مستوى الإتاحة", False)
    review = {}
    for r in rows[h + 1:]:
        rid = _get(r, rr)
        if re.fullmatch(r"R\d{6}", rid):
            review[rid] = {"status": _get(r, rs), "access": _get(r, ra)}

    # ---- سجلات التوثيق ----
    rows = _rows(wb, config.SHEET_RECORDS)
    h, hdr = _header(rows, "رقم الحرفة", config.SHEET_RECORDS)
    c_c = _col(hdr, "رقم الحرفة", sheet=config.SHEET_RECORDS)
    c_r = _col(hdr, "رقم سجل التوثيق", sheet=config.SHEET_RECORDS)
    meta_cols = {
        "governorate": _col(hdr, "المحافظة", False), "center": _col(hdr, "المركز", False),
        "village": _col(hdr, "القرية", False), "collected": _col(hdr, "تاريخ الجمع", False),
        "period": _col(hdr, "الفترة", False), "enterer": _col(hdr, "مُدخل البيانات", False),
        "method": _col(hdr, "طريقة الجمع", False), "notes": _col(hdr, "ملاحظات تجميع", False),
    }
    qcols = {}
    for i, t in enumerate(hdr):
        m = re.match(r"^س\s*(\d+)\s*(.*)$", t)
        if m:
            label = re.sub(r"\s*\(.*?\)\s*$", "", m.group(2)).strip()
            qcols[int(m.group(1))] = (i, label)
    if len(qcols) < 20:
        raise SchemaError(f"لقيت {len(qcols)} عمود أسئلة بس في «{config.SHEET_RECORDS}» (المتوقع ~40). الشكل اتغير؟")

    records, seen = [], set()
    for r in rows[h + 1:]:
        cid, rid = _get(r, c_c), _get(r, c_r)
        if not (re.fullmatch(r"C\d{6}", cid) and re.fullmatch(r"R\d{6}", rid)):
            continue
        if rid in seen:
            warnings.append(f"رقم السجل {rid} متكرر، اتاخد أول ظهور بس")
            continue
        seen.add(rid)
        rec = {"craft_id": cid, "record_id": rid, "raw_answers": {}, "answers": {}}
        for k, i in meta_cols.items():
            rec[k] = _get(r, i)
        for q, (i, _) in qcols.items():
            raw = _get(r, i)
            rec["raw_answers"][q] = raw
            ans = clean_answer(raw)
            if ans:
                rec["answers"][q] = ans
        rec.update({f"review_{k}": v for k, v in review.get(rid, {}).items()})
        records.append(rec)

    wb.close()
    return {
        "crafts": crafts, "sources": sources, "evidence": {k: dict(v) for k, v in evidence.items()},
        "review": review, "records": records, "question_labels": {q: l for q, (_, l) in qcols.items()},
        "warnings": warnings,
    }
