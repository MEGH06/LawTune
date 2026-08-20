"""GraphRAG: vector recall, then graph reasoning over what came back.

Plain vector RAG answers "which passages look like the question". That is the wrong
question for law, where what matters is which *authority* governs, whether it is still
good law, and what else construes the same provision. The graph supplies exactly those
three things, and none of them are recoverable from cosine similarity.

Pipeline:

  1. dense recall   -- FAISS over chunks, multilingual so a Tamil query hits English text
  2. provision hook -- Sections/Articles named in the query are resolved via the graph,
                       which retrieves the governing cases even when wording differs
  3. graph expand   -- precedents cited by the hits (1-2 hops); a case the top hit relies
                       on is usually more authoritative than the hit itself
  4. fuse + rank    -- similarity, citation in-degree (authority), recency, provision match
  5. treatment      -- later judgments that overruled or distinguished a candidate become
                       an explicit warning in the context, not a silent omission
  6. assemble       -- numbered, attributed context the model is told not to go beyond
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .extract import extract_provisions
from .graph import LegalGraph
from .index import Embedder, VectorIndex

# How much each signal contributes to the final ranking.
W_VECTOR = 1.00      # semantic match with the question
W_AUTHORITY = 0.28   # how often the corpus cites this judgment
W_RECENCY = 0.16     # newer law generally governs
W_PROVISION = 0.45   # the query named a Section and this judgment construes it
W_GRAPH_ONLY = 0.55  # discount for candidates the vector search never surfaced

NEGATIVE_TREATMENTS = {"overruled", "reversed", "distinguished", "dissented"}


@dataclass
class Passage:
    chunk_id: str
    cid: str
    text: str
    score: float


@dataclass
class Candidate:
    cid: str
    title: str = ""
    citation: str = ""
    year: int | None = None
    court: str = ""
    disposal: str = ""
    vector_score: float = 0.0
    authority: int = 0
    provision_hits: int = 0
    from_graph: bool = False
    final_score: float = 0.0
    passages: list[Passage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    query: str
    candidates: list[Candidate]
    provisions: list[dict]
    stats: dict

    @property
    def is_empty(self) -> bool:
        return not any(c.passages for c in self.candidates)


class GraphRAG:
    def __init__(self, graph_path: str | Path, index_path: str | Path,
                 embedder: Embedder | None = None, model_name: str | None = None,
                 fallback_embeddings: bool = False):
        self.graph = LegalGraph(graph_path, read_only=True)
        self.index = VectorIndex.load(index_path)
        if embedder is not None:
            self.embedder = embedder
        else:
            # Default to whatever built the index. Querying with a different model than
            # you indexed with silently returns nonsense, so the index records its own.
            self.embedder = Embedder(model_name or self.index.model_name,
                                     fallback=fallback_embeddings,
                                     dim=self.index.dim)
        if not fallback_embeddings and self.embedder.dim != self.index.dim:
            raise SystemExit(
                f"Embedding dim {self.embedder.dim} != index dim {self.index.dim}. "
                f"The index was built with {self.index.model_name!r}; rebuild it or "
                f"query with that model.")
        self._chunk_text: dict[str, str] = {
            m["chunk_id"]: m.get("text", "") for m in self.index.meta}

    def close(self) -> None:
        self.graph.close()

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, k_vec: int = 24, hops: int = 1,
                 k_final: int = 6, max_passages: int = 2) -> Retrieval:
        qvec = self.embedder.encode([query], is_query=True)
        hits = self.index.search(qvec, k=k_vec)[0]

        cands: dict[str, Candidate] = {}
        for h in hits:
            cid = h["cid"]
            c = cands.setdefault(cid, Candidate(cid=cid))
            c.vector_score = max(c.vector_score, h["score"])
            c.passages.append(Passage(h["chunk_id"], cid,
                                      self._chunk_text.get(h["chunk_id"], ""), h["score"]))

        # 2. provisions named in the question -> judgments construing them
        q_provisions = extract_provisions(query)
        for prov in q_provisions:
            for row in self.graph.by_provision(prov["pid"], limit=8):
                c = cands.setdefault(row["cid"], Candidate(cid=row["cid"],
                                                           from_graph=True))
                c.provision_hits += 1

        # 3. precedent expansion from the strongest vector hits
        seeds = [c.cid for c in sorted(cands.values(),
                                       key=lambda x: -x.vector_score)[:6] if c.vector_score]
        for row in self.graph.neighbourhood(seeds, hops=hops, limit=30):
            if row["cid"] not in cands:
                cands[row["cid"]] = Candidate(cid=row["cid"], from_graph=True)

        if not cands:
            return Retrieval(query, [], q_provisions,
                             {"vector_hits": 0, "candidates": 0})

        # 4. hydrate from the graph and fuse the signals
        meta = self.graph.judgments(list(cands))
        authority = {r["cid"]: r["in_degree"] for r in self.graph.most_cited(limit=500)}
        for cid, c in cands.items():
            m = meta.get(cid, {})
            c.title = m.get("title") or ""
            c.citation = m.get("citation") or ""
            c.year = m.get("year")
            c.court = m.get("court") or ""
            c.disposal = m.get("disposal") or ""
            c.authority = authority.get(cid, 0)
            c.final_score = self._score(c)

        # 5. treatment warnings for whatever we are about to show
        ranked = sorted(cands.values(), key=lambda c: -c.final_score)[:k_final]
        self._attach_warnings(ranked)

        for c in ranked:
            c.passages.sort(key=lambda p: -p.score)
            c.passages = [p for p in c.passages[:max_passages] if p.text]

        return Retrieval(query, ranked, q_provisions, {
            "vector_hits": len(hits),
            "candidates": len(cands),
            "from_graph": sum(1 for c in cands.values() if c.from_graph),
            "returned": len(ranked),
        })

    def _score(self, c: Candidate) -> float:
        # log1p keeps a single much-cited case from dominating every answer.
        authority = math.log1p(c.authority) / math.log(50)
        recency = 0.0
        if c.year:
            recency = max(0.0, min(1.0, (c.year - 1990) / 36))
        score = (W_VECTOR * c.vector_score
                 + W_AUTHORITY * authority
                 + W_RECENCY * recency
                 + W_PROVISION * min(c.provision_hits, 2) / 2)
        if c.from_graph and not c.vector_score:
            score *= W_GRAPH_ONLY
        return round(score, 5)

    def _attach_warnings(self, cands: Sequence[Candidate]) -> None:
        """Say so when a later judgment treated this one negatively.

        Retrieval that hands over an overruled case with no flag is worse than no
        retrieval, because it launders a wrong answer through a real citation.
        """
        cids = [c.cid for c in cands]
        by_target: dict[str, list[dict]] = {}
        for row in self.graph.citing(cids, limit=200):
            by_target.setdefault(row["target"], []).append(row)
        for c in cands:
            for row in by_target.get(c.cid, []):
                if row.get("treatment") in NEGATIVE_TREATMENTS:
                    year = f" ({row['year']})" if row.get("year") else ""
                    c.warnings.append(
                        f"{row['treatment']} in {row.get('title') or row['cid']}{year}")
            c.warnings = c.warnings[:3]

    # ------------------------------------------------------------------ prompting
    @staticmethod
    def format_context(r: Retrieval, max_chars: int = 6000) -> str:
        if r.is_empty:
            return ""
        blocks, used = [], 0
        for i, c in enumerate(r.candidates, 1):
            if not c.passages:
                continue
            head = f"[{i}] {c.title or c.cid}"
            if c.citation:
                head += f", {c.citation}"
            if c.year:
                head += f" ({c.year})"
            if c.warnings:
                head += "\n    ⚠ later treatment: " + "; ".join(c.warnings)
            body = "\n".join(f"    {p.text.strip()}" for p in c.passages)
            block = f"{head}\n{body}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    @staticmethod
    def build_question(query: str, context: str) -> str:
        """Wrap the question so the model answers from the context and nothing else."""
        if not context:
            return (
                f"{query}\n\n"
                "[Retrieval note: no judgment in the local corpus matched this question. "
                "Say that you have no retrieved authority for it, answer only from "
                "settled statutory provisions you are sure of, and recommend verifying "
                "on indiacode.nic.in.]"
            )
        return (
            "Answer the question using ONLY the retrieved judgments below.\n"
            "Rules for this answer:\n"
            "- Cite the judgments you rely on by their bracketed number and name.\n"
            "- If the passages do not settle the question, say exactly that. Do not "
            "fill the gap from memory.\n"
            "- If a judgment carries a later-treatment warning, say so before relying "
            "on it.\n"
            "- Answer in the same language as the question.\n\n"
            f"RETRIEVED JUDGMENTS\n{context}\n\n"
            f"QUESTION\n{query}"
        )

    def answer_prompt(self, query: str, **kw) -> tuple[str, Retrieval]:
        r = self.retrieve(query, **kw)
        return self.build_question(query, self.format_context(r)), r
