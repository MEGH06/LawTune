#!/usr/bin/env python
"""Build the legal knowledge graph from fetched judgments.

Two passes, because citation edges need to know every judgment before any of them can
be resolved: pass one loads metadata and builds citation-key -> cid; pass two extracts
facts from the text and writes nodes and edges.

Judgments without text still become nodes. They carry a bench, a date, a court, a
disposal and -- crucially -- inbound citations from judgments that do have text, so
they still participate in authority ranking.

    python scripts/09_build_graph.py
    python scripts/09_build_graph.py --limit 200        # quick build
    python scripts/09_build_graph.py --export-neo4j     # also emit Neo4j CSVs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawtune.config import INTERIM, PROCESSED  # noqa: E402
from lawtune.kg.extract import extract_all  # noqa: E402
from lawtune.kg.graph import LegalGraph, build_resolver  # noqa: E402
from lawtune.kg.schema import AREAS  # noqa: E402

JUDGMENTS = INTERIM / "judgments"
META_IN = JUDGMENTS / "metadata.jsonl"
TEXT_DIR = JUDGMENTS / "text"
GRAPH_OUT = PROCESSED / "legal_graph.kuzu"
FACTS_OUT = PROCESSED / "extracted_facts.jsonl"
REPORT = PROCESSED / "graph_report.json"


def load_metadata(limit: int = 0) -> list[dict]:
    if not META_IN.exists():
        raise SystemExit(f"{META_IN} not found. Run scripts/08_fetch_judgments.py first.")
    rows = [json.loads(l) for l in META_IN.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Judgments with text first, so --limit keeps the useful ones.
    rows.sort(key=lambda r: (not r.get("has_text"), -(r.get("year") or 0)))
    return rows[:limit] if limit else rows


def read_text(rec: dict) -> str:
    path = rec.get("path") or ""
    if not path:
        return ""
    f = TEXT_DIR / f"{path.replace('/', '_')}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(GRAPH_OUT))
    ap.add_argument("--export-neo4j", action="store_true")
    ap.add_argument("--keep", action="store_true", help="add to an existing graph")
    args = ap.parse_args()

    rows = load_metadata(args.limit)
    with_text = sum(1 for r in rows if r.get("has_text"))
    print(f"{len(rows)} judgments ({with_text} with text)\n")

    print("pass 1: citation resolver")
    resolver = build_resolver(rows)
    print(f"  {len(resolver)} citation keys -> {len(set(resolver.values()))} judgments\n")

    print("pass 2: extract and load")
    graph = LegalGraph.create(args.out, overwrite=not args.keep)
    t0 = time.time()
    facts_fh = FACTS_OUT.open("w", encoding="utf-8")
    totals: Counter = Counter()
    area_counter: Counter = Counter()

    for i, rec in enumerate(rows, 1):
        text = read_text(rec)
        facts = extract_all(text, rec.get("title", ""), rec.get("judge"))

        meta = dict(rec)
        meta["source_path"] = rec.get("path", "")
        meta["n_chars"] = len(text)
        meta["has_text"] = bool(text)
        graph.add_judgment(meta, facts)

        if facts["citations"]:
            resolved, stubbed = graph.add_citation_edges(
                rec["cid"], facts["citations"], resolver)
            totals["cites_resolved"] += resolved
            totals["cites_stubbed"] += stubbed

        for k in ("citations", "provisions", "areas", "doctrines", "judges"):
            totals[k] += len(facts[k])
        for a in facts["areas"]:
            area_counter[a["area"]] += 1

        facts_fh.write(json.dumps(
            {"cid": rec["cid"], "n_chars": len(text), **facts},
            ensure_ascii=False) + "\n")

        if i % 100 == 0 or i == len(rows):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(rows)}  {rate:.1f}/s  "
                  f"cites={totals['citations']} provisions={totals['provisions']}")

    facts_fh.close()
    graph.finalise()
    stats = graph.stats()

    if args.export_neo4j:
        out = graph.export_neo4j_csv(PROCESSED / "neo4j_export")
        print(f"\nNeo4j CSVs -> {out}")

    graph.close()

    print("\n" + "=" * 62)
    print("GRAPH")
    for k in ("Judgment", "Judgment_with_text", "Judge", "Provision", "Act",
              "Area", "Doctrine", "Court"):
        print(f"  {k:<22} {stats[k]:>8,}")
    print("EDGES")
    for k in ("CITES", "INTERPRETS", "UNDER_ACT", "DECIDED_BY", "ABOUT",
              "INVOKES", "PART_OF"):
        print(f"  {k:<22} {stats[k]:>8,}")
    print(f"\ncitations resolved to corpus judgments : {totals['cites_resolved']:,}")
    print(f"citations kept as stub precedent nodes  : {totals['cites_stubbed']:,}")

    print("\nAREA COVERAGE (this is the 'everything, not just marriage' check)")
    for key, n in area_counter.most_common():
        print(f"  {AREAS[key][0]:<34} {n:>7,}")
    missing = [AREAS[k][0] for k in AREAS if k not in area_counter]
    if missing:
        print(f"\n  no judgments yet in: {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))
        print("  -> widen --years or raise --max-per-year in 08 to fill these in.")

    REPORT.write_text(json.dumps({
        "nodes_edges": stats, "totals": dict(totals),
        "areas": {AREAS[k][0]: n for k, n in area_counter.most_common()},
        "areas_empty": missing,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngraph  -> {args.out}")
    print(f"facts  -> {FACTS_OUT}")
    print(f"report -> {REPORT}")
    print("\nnext: python scripts/10_build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
