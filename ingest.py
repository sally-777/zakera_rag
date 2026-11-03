"""فهرسة ملف الداتا. شغّله كل ما فريق الداتا يحدّث الملف:
    python ingest.py                  # فهرسة (تزايدية: بتحسب بس الجديد/المعدّل)
    python ingest.py --file path.xlsx
    python ingest.py --report-only    # تقرير جودة الداتا من غير فهرسة
الملف الأصلي مش بيتعدّل أبدًا؛ التنضيف بيحصل في الذاكرة وبيتكتب تقرير في index/quality_report.md"""
import argparse
import json
import logging
import re
import sys
from collections import Counter

import config
from rag import chunker, loader
from rag.index import Index

log = logging.getLogger("zakera.ingest")
SUSPICIOUS = re.compile(r"افتراضي|لتجربة|تجريبي|lorem", re.I)


def quality_report(data, chunks, stats):
    recs, crafts, labels = data["records"], data["crafts"], data["question_labels"]
    enterers = {r.get("enterer") for r in recs if r.get("enterer")}
    n = max(len(recs), 1)
    fill = {q: sum(1 for r in recs if q in r["answers"]) for q in sorted(labels)}
    issues = []
    for cid, c in crafts.items():
        if c["name"] in enterers:
            issues.append(f"{cid}: اسم الحرفة «{c['name']}» هو اسم مُدخل بيانات مش اسم حرفة (غالبًا خطأ إدخال في «دليل الحرف»)")
        elif not c["description"]:
            issues.append(f"{cid} ({c['name']}): مفيش وصف تعريفي")
    used = {r["craft_id"] for r in recs}
    for cid, c in crafts.items():
        if cid not in used:
            issues.append(f"{cid} ({c['name']}): مفيش ليها أي سجل توثيق")
    for r in recs:
        for q, raw in r["raw_answers"].items():
            if raw and SUSPICIOUS.search(raw[:200]):
                issues.append(f"{r['record_id']} س{q}: فيه نص شكله تجريبي/افتراضي -> «{raw[:60]}…»")
        ans_q = {q for q in r["answers"] if q not in config.EXCLUDE_QUESTIONS}
        ev_q = set(data["evidence"].get(r["record_id"], {}))
        if ans_q and not ev_q:
            issues.append(f"{r['record_id']}: مفيهوش أي دليل/مصدر مربوط في شيت «الأدلة»")
    return {
        "records": len(recs), "crafts": len(crafts), "sources": len(data["sources"]), "chunks": len(chunks),
        "status_counts": dict(Counter(r.get("review_status") or "بدون حالة" for r in recs)),
        "governorates": dict(Counter(r.get("governorate") or "غير محددة" for r in recs)),
        "fill_rate_per_question": {f"س{q} {labels[q]}": f"{fill[q]}/{len(recs)}" for q in fill},
        "chunk_stats": stats, "loader_warnings": data["warnings"], "issues": issues,
    }


def write_report(rep):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / "quality_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# تقرير جودة الداتا\n",
          f"- السجلات: {rep['records']} | الحرف: {rep['crafts']} | المصادر: {rep['sources']} | القطع المفهرسة: {rep['chunks']}",
          f"- حالات السجلات: {rep['status_counts']}", f"- المحافظات: {rep['governorates']}\n",
          "## مشاكل محتاجة مراجعة من فريق الداتا"] + [f"- {i}" for i in rep["issues"]] + \
         ["\n## نسبة الإجابات الفعلية لكل سؤال (بعد حذف «غير مذكور/غير معروف»)"] + \
         [f"- {k}: {v}" for k, v in rep["fill_rate_per_question"].items()]
    (config.INDEX_DIR / "quality_report.md").write_text("\n".join(md), encoding="utf-8")


def run(path=None, report_only=False):
    path = path or config.DATA_FILE
    data = loader.load(path)
    chunks, stats = chunker.build_chunks(data)
    rep = quality_report(data, chunks, stats)
    write_report(rep)
    if report_only:
        return rep
    if not chunks:
        raise SystemExit("مفيش أي قطع للفهرسة. راجع ALLOWED_STATUS في الإعدادات (الحالات الموجودة: %s)" % rep["status_counts"])
    Index.build(chunks, previous=Index.load()).save()
    return rep


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    try:
        rep = run(a.file and __import__("pathlib").Path(a.file), a.report_only)
    except loader.SchemaError as e:
        sys.exit(f"❌ شكل الملف اتغير: {e}")
    print(f"✅ سجلات: {rep['records']} | حرف: {rep['crafts']} | قطع مفهرسة: {rep['chunks']} | مشاكل في الداتا: {len(rep['issues'])}")
    print(f"   التقرير: {config.INDEX_DIR / 'quality_report.md'}")
