#!/usr/bin/env python
"""Chunk the judgments, embed them, build the FAISS index, and wire it into the graph.

The Chunk nodes written here are the join between the two stores: FAISS returns a row
number, the graph turns that row into a judgment, and from there into precedents,
provisions and later treatment. Without this step you have a vector database and a graph
that know nothing about each other.

Embeddings are multilingual by default so a Tamil or Bengali question retrieves English
judgments -- which is the entire point of pairing this with a 23-language model.

    python scripts/10_build_index.py
    python scripts/10_build_index.py --model BAAI/bge-m3          # better, heavier
    python scripts/10_build_index.py --fallback-embeddings        # no model download
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from lawtune.config import INTERIM, PROCESSED  # noqa: E402
from lawtune.kg.graph import LegalGraph  # noqa: E402
from lawtune.kg.index import DEFAULT_MODEL, Embedder, VectorIndex, chunk_records  # noqa: E402

JUDGMENTS = INTERIM / "judgments"
META_IN = JUDGMENTS / "metadata.jsonl"
TEXT_DIR = JUDGMENTS / "text"
GRAPH = PROCESSED / "legal_graph.kuzu"
INDEX_OUT = PROCESSED / "faiss_index"
REPORT = PROCESSED / "index_report.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--graph", type=str, default=str(GRAPH))
    ap.add_argument("--out", type=str, default=str(INDEX_OUT))
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--fallback-embeddings", action="store_true",
                    help="hashed embeddings, no model download (testing only)")
    args = ap.parse_args()

    if not META_IN.exists():
        raise SystemExit(f"{META_IN} not found. Run scripts/08_fetch_judgments.py.")
    if not Path(args.graph).exists():
        raise SystemExit(f"{args.graph} not found. Run scripts/09_build_graph.py.")

    rows = [json.loads(l) for l in META_IN.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("has_text")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(
            "No judgments with text. Re-run 08_fetch_judgments.py without "
            "--metadata-only: retrieval needs the text, the graph alone is not enough.")
    print(f"{len(rows)} judgments with text\n")

    print("chunking ...")
    all_chunks: list[dict] = []
    for rec in rows:
        path = (rec.get("path") or "").replace("/", "_")
        f = TEXT_DIR / f"{path}.txt"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        all_chunks += chunk_records(rec["cid"], text, start_row=len(all_chunks),
                                    target=args.chunk_size, overlap=args.overlap)
    if not all_chunks:
        raise SystemExit("Chunking produced nothing -- check data/interim/judgments/text/.")
    sizes = sorted(c["n_chars"] for c in all_chunks)
    print(f"  {len(all_chunks):,} chunks   median {sizes[len(sizes)//2]} chars   "
          f"max {sizes[-1]}\n")

    embedder = Embedder(args.model, device=args.device or None,
                        fallback=args.fallback_embeddings)
    print(f"embedding {len(all_chunks):,} chunks (dim {embedder.dim}) ...")
    t0 = time.time()
    vectors = embedder.encode([c["text"] for c in all_chunks],
                              batch_size=args.batch_size, show_progress=True)
    dt = time.time() - t0
    print(f"  {dt:.0f}s  ({len(all_chunks)/max(dt,1e-9):.0f} chunks/s)\n")

    meta = [{"chunk_id": c["chunk_id"], "cid": c["cid"], "ordinal": c["ordinal"],
             "n_chars": c["n_chars"], "text": c["text"]} for c in all_chunks]
    index = VectorIndex.build(np.asarray(vectors), meta,
                              model_name="hash-fallback" if args.fallback_embeddings
                              else args.model)
    index.save(args.out)
    print(f"FAISS index -> {args.out}  ({index.index.ntotal:,} vectors, "
          f"{type(index.index).__name__})")

    # Wire the chunks into the graph so retrieval can walk vector hits into precedent.
    print("\nlinking chunks into the graph ...")
    graph = LegalGraph(args.graph)
    by_cid: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_cid.setdefault(c["cid"], []).append(c)
    for i, (cid, chunks) in enumerate(by_cid.items(), 1):
        graph.add_chunks(cid, chunks)
        if i % 200 == 0:
            print(f"  {i}/{len(by_cid)} judgments linked")
    stats = graph.stats()
    graph.close()
    print(f"  {stats['Chunk']:,} Chunk nodes, {stats['HAS_CHUNK']:,} HAS_CHUNK edges")

    REPORT.write_text(json.dumps({
        "judgments": len(rows), "chunks": len(all_chunks),
        "model": args.model if not args.fallback_embeddings else "hash-fallback",
        "dim": embedder.dim, "seconds": round(dt, 1),
        "median_chunk_chars": sizes[len(sizes) // 2],
    }, indent=2), encoding="utf-8")

    if args.fallback_embeddings:
        print("\nNOTE: hashed fallback embeddings. The pipeline is proven end to end, "
              "but retrieval quality is poor. Re-run without --fallback-embeddings "
              "before judging results.")
    print("\nnext: python scripts/11_ask.py \"What is anticipatory bail?\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
