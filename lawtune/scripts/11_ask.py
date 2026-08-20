#!/usr/bin/env python
"""Query the GraphRAG stack. Inspect retrieval, or get a grounded answer.

    python scripts/11_ask.py "What is anticipatory bail?"
    python scripts/11_ask.py "धारा 302 क्या कहती है?" --show-context
    python scripts/11_ask.py "cheque bounce" --retrieval-only     # no model needed
    python scripts/11_ask.py --explore                            # graph statistics

--retrieval-only is the one to reach for while tuning: it exercises the whole retrieval
path without loading a 2B model, so you can see exactly what the model would have been
given and decide whether the answer's problem is retrieval or generation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawtune.config import MAX_SEQ_LEN, PROCESSED  # noqa: E402
from lawtune.kg.graph import LegalGraph  # noqa: E402
from lawtune.kg.retrieve import GraphRAG  # noqa: E402

GRAPH = PROCESSED / "legal_graph.kuzu"
INDEX = PROCESSED / "faiss_index"


def explore(graph_path: str) -> int:
    g = LegalGraph(graph_path, read_only=True)
    s = g.stats()
    print("NODES")
    for k in ("Judgment", "Judgment_with_text", "Judge", "Provision", "Act",
              "Area", "Doctrine", "Chunk"):
        print(f"  {k:<22} {s[k]:>8,}")
    print("EDGES")
    for k in ("CITES", "INTERPRETS", "UNDER_ACT", "DECIDED_BY", "ABOUT", "INVOKES"):
        print(f"  {k:<22} {s[k]:>8,}")

    print("\nMOST-CITED JUDGMENTS")
    for r in g.most_cited(12):
        title = (r["title"] or r["cid"])[:58]
        print(f"  {r['in_degree']:>4}  {title:<60} {r.get('citation') or ''}")

    print("\nAREA COVERAGE")
    for r in g.area_coverage()[:40]:
        print(f"  {r['name']:<34} {r['n']:>7,}")
    g.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question to ask")
    ap.add_argument("--graph", type=str, default=str(GRAPH))
    ap.add_argument("--index", type=str, default=str(INDEX))
    ap.add_argument("--model", type=str, default="", help="LawTune adapter or model path")
    ap.add_argument("--k-vec", type=int, default=24)
    ap.add_argument("--k-final", type=int, default=6)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--fallback-embeddings", action="store_true")
    ap.add_argument("--explore", action="store_true")
    args = ap.parse_args()

    if args.explore:
        return explore(args.graph)

    question = " ".join(args.question).strip()
    if not question:
        ap.error("give a question, or pass --explore")

    for p, hint in ((args.graph, "09_build_graph.py"), (args.index, "10_build_index.py")):
        if not Path(p).exists():
            raise SystemExit(f"{p} not found. Run scripts/{hint} first.")

    rag = GraphRAG(args.graph, args.index,
                   fallback_embeddings=args.fallback_embeddings)
    prompt, r = rag.answer_prompt(question, k_vec=args.k_vec, hops=args.hops,
                                  k_final=args.k_final)

    print(f"QUESTION  {question}")
    if r.provisions:
        print(f"provisions detected in the question: "
              f"{[p['label'] for p in r.provisions]}")
    print(f"retrieval: {r.stats}\n")

    print("RETRIEVED")
    for i, c in enumerate(r.candidates, 1):
        flag = " [graph-only]" if c.from_graph and not c.vector_score else ""
        print(f"  [{i}] {(c.title or c.cid)[:64]}{flag}")
        print(f"      score={c.final_score:.3f} vec={c.vector_score:.3f} "
              f"authority={c.authority} prov={c.provision_hits} year={c.year}")
        if c.citation:
            print(f"      {c.citation}")
        for w in c.warnings:
            print(f"      /!\\ {w}")

    if args.show_context or args.retrieval_only:
        print("\nCONTEXT GIVEN TO THE MODEL\n" + "-" * 62)
        print(rag.format_context(r) or "(nothing retrieved)")
        print("-" * 62)

    if args.retrieval_only:
        rag.close()
        return 0

    from lawtune.guardrails import check_input, check_output
    from lawtune.model import LawTuneModel

    verdict = check_input(question)
    if not verdict.allowed:
        print("\nANSWER (blocked before generation)\n" + verdict.replacement)
        rag.close()
        return 0

    print("\nloading model ...")
    bot = LawTuneModel(args.model, MAX_SEQ_LEN)
    print(f"model: {bot.model_id}\n\nANSWER")
    raw = bot.ask(prompt, max_new_tokens=args.max_new_tokens,
                  temperature=args.temperature, stream=True)
    out = check_output(raw, question)
    if not out.allowed:
        print("\n\n[guardrail] withheld:\n" + (out.replacement or ""))
    elif out.flags:
        print(f"\n[guardrail flags: {', '.join(out.flags)}]")
    rag.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
