#!/usr/bin/env python
"""Fetch Supreme Court judgments from the AWS Open Data mirror of eCourts.

Source: s3://indian-supreme-court-judgments (ap-south-1), CC-BY-4.0, anonymous access,
1950-2025, refreshed bi-monthly. This is the free, legitimate route: no Indian Kanoon
scraping, no CAPTCHA, no API key, no terms-of-service problem.

    metadata/parquet/year=YYYY/metadata.parquet   ~1 MB/year, fully structured
    data/pdf/year=YYYY/english/<path>_EN.pdf      one PDF per judgment

Two things worth knowing:

  * Metadata is cheap and complete, PDFs are not. So metadata for the whole window is
    always fetched; text is fetched for as many judgments as you allow. The graph is
    built from metadata and degrades gracefully when text is missing -- a judgment with
    no text is still a node with a bench, a date, a disposal and inbound citations.
  * Everything is resumable. Downloaded PDFs and extracted text are cached on disk, so
    re-running costs nothing and an interrupted run loses only the file in flight.

    python scripts/08_fetch_judgments.py                          # 2020-2025, 400/yr
    python scripts/08_fetch_judgments.py --years 2020-2025 --max-per-year 1000
    python scripts/08_fetch_judgments.py --metadata-only          # graph without text
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawtune.config import INTERIM  # noqa: E402

BUCKET = "indian-supreme-court-judgments"
BASE = f"https://{BUCKET}.s3.ap-south-1.amazonaws.com/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

OUT = INTERIM / "judgments"
PDF_CACHE = OUT / "pdf"
TEXT_CACHE = OUT / "text"
META_OUT = OUT / "metadata.jsonl"
REPORT = OUT / "fetch_report.json"


def http_get(url: str, timeout: int = 90, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LawTune/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def s3_list(prefix: str, max_keys: int = 1000) -> list[str]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(http_get(BASE + "?" + urllib.parse.urlencode(q)))
        keys += [c.findtext("s3:Key", "", NS) for c in root.findall("s3:Contents", NS)]
        if root.findtext("s3:IsTruncated", "false", NS) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", None, NS)
        if not token:
            return keys


def parse_years(spec: str) -> list[int]:
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            years += list(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def parse_date(raw: str | None) -> str | None:
    """The mirror writes DD-MM-YYYY; the graph wants ISO."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def load_year_metadata(year: int) -> list[dict]:
    import pyarrow.parquet as pq

    key = f"metadata/parquet/year={year}/metadata.parquet"
    try:
        raw = http_get(BASE + key)
    except RuntimeError as exc:
        print(f"    ! no metadata for {year}: {exc}")
        return []
    table = pq.read_table(io.BytesIO(raw))
    rows = []
    for r in table.to_pylist():
        case_id = (r.get("case_id") or "").strip()
        cnr = (r.get("cnr") or "").strip()
        path = (r.get("path") or "").strip()
        cid = case_id or cnr or path
        if not cid:
            continue
        rows.append({
            "cid": cid,
            "title": (r.get("title") or "").strip(),
            "petitioner": (r.get("petitioner") or "").strip(),
            "respondent": (r.get("respondent") or "").strip(),
            "citation": (r.get("citation") or "").strip(),
            # case_id on this mirror IS the neutral citation ("2023 INSC 1043")
            "neutral_citation": case_id,
            "judge": (r.get("judge") or "").strip(),
            "author_judge": (r.get("author_judge") or "").strip(),
            "decision_date": parse_date(r.get("decision_date")),
            "disposal": (r.get("disposal_nature") or "").strip(),
            "court": (r.get("court") or "Supreme Court of India").strip(),
            "languages": (r.get("available_languages") or "").strip(),
            "path": path,
            "year": int(r.get("year") or year),
        })
    return rows


def pdf_url(year: int, path: str) -> str:
    return f"{BASE}data/pdf/year={year}/english/{path}_EN.pdf"


def extract_pdf_text(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def fetch_one(rec: dict, keep_pdf: bool) -> tuple[str, int, str]:
    """Return (cid, n_chars, status). Cached results cost nothing."""
    cid, path, year = rec["cid"], rec["path"], rec["year"]
    if not path:
        return cid, 0, "no_path"

    safe = path.replace("/", "_")
    text_file = TEXT_CACHE / f"{safe}.txt"
    if text_file.exists():
        return cid, len(text_file.read_text(encoding="utf-8")), "cached"

    try:
        pdf_file = PDF_CACHE / f"{safe}.pdf"
        raw = pdf_file.read_bytes() if pdf_file.exists() else http_get(pdf_url(year, path))
        if keep_pdf and not pdf_file.exists():
            pdf_file.write_bytes(raw)
        text = extract_pdf_text(raw)
    except Exception as exc:                                     # noqa: BLE001
        return cid, 0, f"error:{type(exc).__name__}"

    if len(text.strip()) < 500:
        return cid, len(text), "too_short"
    text_file.write_text(text, encoding="utf-8")
    return cid, len(text), "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default="2020-2025",
                    help="e.g. 2020-2025 or 2021,2023")
    ap.add_argument("--max-per-year", type=int, default=400,
                    help="cap on judgments whose TEXT is downloaded (0 = all)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="build the graph from structured metadata, skip PDFs")
    ap.add_argument("--keep-pdf", action="store_true", help="cache the PDFs too")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    for d in (OUT, PDF_CACHE, TEXT_CACHE):
        d.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.years)
    print(f"Supreme Court judgments, years {years[0]}-{years[-1]} "
          f"({len(years)} years) from s3://{BUCKET}\n")

    all_meta: list[dict] = []
    report: list[dict] = []

    for year in years:
        print(f"[{year}] metadata ... ", end="", flush=True)
        rows = load_year_metadata(year)
        print(f"{len(rows)} judgments")
        if not rows:
            report.append({"year": year, "metadata": 0, "text": 0})
            continue

        targets = rows if not args.max_per_year else rows[: args.max_per_year]
        stats = {"ok": 0, "cached": 0, "too_short": 0, "no_path": 0, "error": 0}

        if not args.metadata_only:
            t0 = time.time()
            done = 0
            with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs = {pool.submit(fetch_one, r, args.keep_pdf): r for r in targets}
                for fut in futures.as_completed(jobs):
                    rec = jobs[fut]
                    cid, n_chars, status = fut.result()
                    rec["n_chars"] = n_chars
                    rec["has_text"] = status in ("ok", "cached")
                    stats[status.split(":")[0] if status.startswith("error") else status] = \
                        stats.get(status.split(":")[0] if status.startswith("error")
                                  else status, 0) + 1
                    done += 1
                    if done % 50 == 0:
                        rate = done / max(time.time() - t0, 1e-9)
                        print(f"    {done}/{len(targets)}  {rate:.1f}/s  "
                              f"ok={stats['ok'] + stats['cached']}")
            print(f"    text: {stats['ok']} new, {stats['cached']} cached, "
                  f"{stats['too_short']} too short, {stats.get('error', 0)} errors "
                  f"({time.time() - t0:.0f}s)")
        else:
            for r in targets:
                r["n_chars"], r["has_text"] = 0, False

        for r in rows:
            r.setdefault("n_chars", 0)
            r.setdefault("has_text", False)
        all_meta += rows
        report.append({"year": year, "metadata": len(rows),
                       "text_attempted": len(targets),
                       "text_ok": stats["ok"] + stats["cached"]})

    with META_OUT.open("w", encoding="utf-8") as fh:
        for r in all_meta:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_text = sum(1 for r in all_meta if r["has_text"])
    REPORT.write_text(json.dumps(
        {"years": years, "total": len(all_meta), "with_text": with_text,
         "per_year": report}, indent=2), encoding="utf-8")

    print(f"\n{len(all_meta)} judgments in metadata, {with_text} with full text")
    print(f"metadata -> {META_OUT}")
    print(f"text     -> {TEXT_CACHE}")
    if not with_text and not args.metadata_only:
        print("\nWARNING: no text downloaded. The graph will still build from metadata, "
              "but retrieval needs text -- check your connection and re-run.")
    print("\nnext: python scripts/09_build_graph.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
