"""Chunking, embeddings and the FAISS index.

Embeddings must be multilingual, because a Tamil question has to retrieve an English
judgment -- that is the whole point of pairing this with a 23-language model. The default
is `intfloat/multilingual-e5-small`: 118M params, 384 dims, ~470 MB, covers 100 languages,
runs on MPS. `bge-m3` is better and much heavier; it is one flag away.

Everything here is free and local. No embedding API, no vector-DB service: FAISS is a
file on disk next to the graph.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

# e5 models are trained with these prefixes and lose accuracy without them.
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥۔])\s+")


# ------------------------------------------------------------------ chunking
def chunk_text(text: str, target: int = 1200, overlap: int = 180,
               min_chars: int = 200) -> list[str]:
    """Paragraph-aware chunks with overlap.

    Judgments argue across paragraph boundaries, so a hard character split routinely
    severs a holding from the reasoning that qualifies it. We pack whole paragraphs up
    to `target`, and only fall back to sentence splitting for paragraphs that are
    themselves oversized.
    """
    text = (text or "").strip()
    if not text:
        return []

    units: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target:
            units.append(para)
            continue
        buf = ""
        for sent in _SENT_SPLIT.split(para):
            if len(buf) + len(sent) + 1 <= target:
                buf = f"{buf} {sent}".strip()
            else:
                if buf:
                    units.append(buf)
                buf = sent[:target] if len(sent) > target else sent
        if buf:
            units.append(buf)

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + len(unit) + 2 <= target:
            buf = f"{buf}\n\n{unit}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{unit}".strip() if tail else unit
    if buf:
        chunks.append(buf)

    out = [c for c in chunks if len(c) >= min_chars]
    return out or ([text[:target]] if len(text) >= min_chars else [])


# ------------------------------------------------------------------ embeddings
class Embedder:
    """sentence-transformers wrapper with an offline fallback.

    The fallback is a deterministic hashed bag-of-character-ngrams. It is NOT good
    retrieval -- it exists so the pipeline, the graph joins and the tests can run with no
    model download and no network. Anything real should use the transformer.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 fallback: bool = False, dim: int = 384):
        self.model_name = model_name
        self.is_e5 = "e5" in model_name.lower()
        self.fallback = fallback
        self.dim = dim
        self.model = None
        if fallback:
            print(f"[embed] hashing fallback, dim={dim} (not for production retrieval)")
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required for real embeddings:\n"
                "    pip install sentence-transformers\n"
                "or pass --fallback-embeddings to run the pipeline without a model.\n"
                f"({type(exc).__name__}: {exc})") from exc
        if device is None:
            device = self._best_device()
        print(f"[embed] loading {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    @staticmethod
    def _best_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _hash_embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            low = " ".join(t.lower().split())
            for n in (3, 4, 5):
                for j in range(max(0, len(low) - n + 1)):
                    h = hashlib.blake2b(low[j:j + n].encode("utf-8"),
                                        digest_size=4).digest()
                    out[i, int.from_bytes(h, "big") % self.dim] += 1.0
        return out

    def encode(self, texts: Sequence[str], is_query: bool = False,
               batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if self.fallback:
            vecs = self._hash_embed(texts)
        else:
            payload = texts
            if self.is_e5:
                prefix = E5_QUERY_PREFIX if is_query else E5_DOC_PREFIX
                payload = [prefix + t for t in texts]
            vecs = self.model.encode(payload, batch_size=batch_size,
                                     show_progress_bar=show_progress,
                                     convert_to_numpy=True).astype("float32")
        # Normalise so inner product == cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


# ------------------------------------------------------------------ FAISS
class VectorIndex:
    """Flat inner-product index below ~200k vectors, HNSW above.

    Flat is exact and needs no training; at this corpus size the recall gain is worth
    far more than the millisecond it costs.
    """

    HNSW_THRESHOLD = 200_000

    def __init__(self, dim: int, index=None, meta: list[dict] | None = None,
                 model_name: str = DEFAULT_MODEL):
        self.dim = dim
        self.index = index
        self.meta: list[dict] = meta or []
        self.model_name = model_name

    @classmethod
    def build(cls, vectors: np.ndarray, meta: list[dict],
              model_name: str = DEFAULT_MODEL) -> "VectorIndex":
        import faiss

        dim = vectors.shape[1]
        if len(vectors) >= cls.HNSW_THRESHOLD:
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(dim, index, meta, model_name)

    def search(self, query_vecs: np.ndarray, k: int = 10) -> list[list[dict]]:
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(query_vecs))]
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vecs.astype("float32"), k)
        out = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            for score, idx in zip(row_scores, row_ids):
                if idx < 0:
                    continue
                hit = dict(self.meta[idx])
                hit["score"] = float(score)
                hit["row"] = int(idx)
                hits.append(hit)
            out.append(hits)
        return out

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "vectors.faiss"))
        with (path / "meta.jsonl").open("w", encoding="utf-8") as fh:
            for m in self.meta:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        (path / "index_config.json").write_text(json.dumps({
            "dim": self.dim, "model_name": self.model_name,
            "n_vectors": int(self.index.ntotal),
            "index_type": type(self.index).__name__,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        import faiss

        path = Path(path)
        cfg_path = path / "index_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"No FAISS index at {path}. Run scripts/10_build_index.py.")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(path / "vectors.faiss"))
        meta = [json.loads(l) for l in
                (path / "meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls(cfg["dim"], index, meta, cfg.get("model_name", DEFAULT_MODEL))


def chunk_records(cid: str, text: str, start_row: int, **kwargs) -> list[dict]:
    """Chunk one judgment into index-ready records."""
    return [
        {"chunk_id": f"{cid}#{i}", "cid": cid, "ordinal": i, "text": c,
         "n_chars": len(c), "vector_row": start_row + i}
        for i, c in enumerate(chunk_text(text, **kwargs))
    ]
