"""The Kuzu-backed legal knowledge graph: build it, then traverse it.

Why Kuzu and not Neo4j: Neo4j Community is free but is a *server* -- a JVM, a service to
start, ports, credentials, and a second thing that can be broken on someone's laptop.
Kuzu is embedded (the database is a directory), speaks Cypher, is MIT-licensed, and
installs with `pip install kuzu`. For a graph you ship inside an app, that is the whole
argument. `export_neo4j_csv()` is provided for anyone who wants the Neo4j browser.

The one non-obvious piece here is citation resolution. A judgment cites cases that are
not in our corpus, and those citations are exactly what carry authority signal. So every
cited case becomes a node: a real one if we ingested it, otherwise a stub keyed by the
normalised citation. Precedent centrality then works even over a partial corpus.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from ..acts import BY_KEY
from .schema import AREAS, DOCTRINES, PARENTS, ddl

CITATION_IN_META = [
    (re.compile(r"\b((?:19|20)\d{2})\s+INSC\s+(\d{1,5})\b", re.I),
     lambda m: f"insc:{m.group(1)}:{m.group(2)}"),
    (re.compile(r"\(?\b((?:19|20)\d{2})\)?\s*\(?(\d{1,2})\)?\s*SCC\s+(\d{1,5})\b"),
     lambda m: f"scc:{m.group(1)}:{m.group(2)}:{m.group(3)}"),
    (re.compile(r"\bAIR\s+((?:19|20)\d{2})\s+([A-Z]{2,4})\s+(\d{1,5})\b"),
     lambda m: f"air:{m.group(1)}:{m.group(2).lower()}:{m.group(3)}"),
    (re.compile(r"\[?\b((?:19|20)\d{2})\]?\s*(\d{1,2})\s*S\.?\s?C\.?\s?R\.?\s+(\d{1,5})\b"),
     lambda m: f"scr:{m.group(1)}:{m.group(2)}:{m.group(3)}"),
]


def citation_keys(raw: str | None) -> list[str]:
    """Every normalised citation key present in a judgment's own citation field.

    One judgment usually carries several parallel citations (SCC, AIR, SCR). Indexing
    all of them is what lets an incoming "AIR 1973 SC 1461" resolve to the same node as
    "(1973) 4 SCC 225".
    """
    if not raw or not isinstance(raw, str):
        return []
    keys = []
    for pattern, fmt in CITATION_IN_META:
        for m in pattern.finditer(raw):
            keys.append(fmt(m))
    return list(dict.fromkeys(keys))


class LegalGraph:
    def __init__(self, path: str | Path, read_only: bool = False):
        import kuzu

        self.path = Path(path)
        self.db = kuzu.Database(str(self.path), read_only=read_only)
        self.conn = kuzu.Connection(self.db)

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, path: str | Path, overwrite: bool = False) -> "LegalGraph":
        path = Path(path)
        if overwrite and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        g = cls(path)
        for stmt in ddl():
            g.conn.execute(stmt)
        g._seed_reference_data()
        return g

    def _seed_reference_data(self) -> None:
        """Acts, provisions' parent Acts, the area taxonomy and doctrines.

        These are known in advance, so the graph is never empty and area/doctrine edges
        always have somewhere to land.
        """
        for act in BY_KEY.values():
            self.conn.execute(
                "MERGE (a:Act {key: $k}) SET a.name = $n, a.year = $y, "
                "a.max_provision = $m, a.category = $c",
                {"k": act.key, "n": act.name, "y": act.year or 0,
                 "m": act.max_provision or 0, "c": act.category})
        for key, name in PARENTS.items():
            self.conn.execute("MERGE (a:Area {key: $k}) SET a.name = $n, a.parent = ''",
                              {"k": key, "n": name})
        for key, (name, parent, _) in AREAS.items():
            self.conn.execute("MERGE (a:Area {key: $k}) SET a.name = $n, a.parent = $p",
                              {"k": key, "n": name, "p": parent})
        for key in DOCTRINES:
            self.conn.execute("MERGE (d:Doctrine {key: $k}) SET d.name = $n",
                              {"k": key, "n": key.replace("_", " ").title()})

    def close(self) -> None:
        self.conn.close()
        self.db.close()

    # ------------------------------------------------------------------ ingestion
    def add_judgment(self, meta: dict, facts: dict) -> None:
        """One judgment plus everything extracted from it."""
        cid = meta["cid"]
        self.conn.execute(
            """MERGE (j:Judgment {cid: $cid})
               SET j.title = $title, j.petitioner = $pet, j.respondent = $res,
                   j.citation = $cit, j.neutral_citation = $ncit, j.year = $year,
                   j.disposal = $disp, j.court = $court, j.bench_size = $bench,
                   j.languages = $langs, j.source_path = $path, j.n_chars = $nchars,
                   j.has_text = $hastext""",
            {"cid": cid, "title": meta.get("title", "") or "",
             "pet": meta.get("petitioner", "") or "", "res": meta.get("respondent", "") or "",
             "cit": meta.get("citation", "") or "", "ncit": meta.get("neutral_citation", "") or "",
             "year": int(meta.get("year") or 0), "disp": meta.get("disposal", "") or "",
             "court": meta.get("court", "Supreme Court of India") or "",
             "bench": len(facts.get("judges", [])),
             "langs": meta.get("languages", "") or "",
             "path": meta.get("source_path", "") or "",
             "nchars": int(meta.get("n_chars") or 0),
             "hastext": bool(meta.get("has_text"))})

        if meta.get("decision_date"):
            self.conn.execute(
                "MATCH (j:Judgment {cid: $cid}) SET j.decision_date = date($d)",
                {"cid": cid, "d": str(meta["decision_date"])[:10]})

        court = meta.get("court") or "Supreme Court of India"
        self.conn.execute("MERGE (c:Court {name: $n}) SET c.level = $l",
                          {"n": court, "l": "apex" if "Supreme" in court else "high"})
        self.conn.execute("MATCH (j:Judgment {cid:$cid}), (c:Court {name:$n}) "
                          "MERGE (j)-[:IN_COURT]->(c)", {"cid": cid, "n": court})

        author = (meta.get("author_judge") or "").strip().lower()
        for name in facts.get("judges", []):
            self.conn.execute("MERGE (p:Judge {name: $n})", {"n": name})
            self.conn.execute(
                "MATCH (j:Judgment {cid:$cid}), (p:Judge {name:$n}) "
                "MERGE (j)-[r:DECIDED_BY]->(p) SET r.authored = $a",
                {"cid": cid, "n": name, "a": bool(author and name.lower() in author)})

        for prov in facts.get("provisions", []):
            self.conn.execute(
                "MERGE (p:Provision {pid: $pid}) SET p.act_key=$ak, p.kind=$k, "
                "p.number=$n, p.suffix=$s, p.label=$l",
                {"pid": prov["pid"], "ak": prov["act_key"], "k": prov["kind"],
                 "n": prov["number"], "s": prov["suffix"], "l": prov["label"]})
            self.conn.execute(
                "MATCH (p:Provision {pid:$pid}), (a:Act {key:$ak}) MERGE (p)-[:PART_OF]->(a)",
                {"pid": prov["pid"], "ak": prov["act_key"]})
            self.conn.execute(
                "MATCH (j:Judgment {cid:$cid}), (p:Provision {pid:$pid}) "
                "MERGE (j)-[r:INTERPRETS]->(p) SET r.mentions = $m",
                {"cid": cid, "pid": prov["pid"], "m": prov["mentions"]})

        for act in facts.get("acts", []):
            self.conn.execute(
                "MATCH (j:Judgment {cid:$cid}), (a:Act {key:$ak}) "
                "MERGE (j)-[r:UNDER_ACT]->(a) SET r.mentions = $m",
                {"cid": cid, "ak": act["act_key"], "m": act["mentions"]})

        for area in facts.get("areas", []):
            self.conn.execute(
                "MATCH (j:Judgment {cid:$cid}), (a:Area {key:$ak}) "
                "MERGE (j)-[r:ABOUT]->(a) SET r.score = $s",
                {"cid": cid, "ak": area["area"], "s": float(area["score"])})

        for doc in facts.get("doctrines", []):
            self.conn.execute(
                "MATCH (j:Judgment {cid:$cid}), (d:Doctrine {key:$dk}) "
                "MERGE (j)-[r:INVOKES]->(d) SET r.mentions = $m",
                {"cid": cid, "dk": doc["doctrine"], "m": doc["mentions"]})

    def add_citation_edges(self, cid: str, citations: Sequence[dict],
                           resolver: dict[str, str]) -> tuple[int, int]:
        """CITES edges. Unresolved citations become stub Judgment nodes.

        Dropping them instead would throw away the precedent signal for every case older
        than the corpus window -- which, for a 2020+ corpus, is most of Indian law.
        """
        resolved = stubbed = 0
        for c in citations:
            target = resolver.get(c["key"])
            if target is None:
                target = f"cite:{c['key']}"
                self.conn.execute(
                    "MERGE (j:Judgment {cid: $cid}) SET j.title = $t, j.citation = $c, "
                    "j.has_text = false, j.court = 'Supreme Court of India'",
                    {"cid": target, "t": c["raw"], "c": c["raw"]})
                stubbed += 1
            else:
                resolved += 1
            if target == cid:
                continue                     # a judgment does not cite itself
            self.conn.execute(
                "MATCH (a:Judgment {cid:$a}), (b:Judgment {cid:$b}) "
                "MERGE (a)-[r:CITES]->(b) SET r.raw = $raw, r.treatment = $t",
                {"a": cid, "b": target, "raw": c["raw"], "t": c["treatment"]})
        return resolved, stubbed

    def add_chunks(self, cid: str, chunks: Sequence[dict]) -> None:
        for ch in chunks:
            self.conn.execute(
                "MERGE (c:Chunk {chunk_id: $id}) SET c.cid=$cid, c.ordinal=$o, "
                "c.n_chars=$n, c.vector_row=$v",
                {"id": ch["chunk_id"], "cid": cid, "o": ch["ordinal"],
                 "n": ch["n_chars"], "v": ch["vector_row"]})
            self.conn.execute(
                "MATCH (j:Judgment {cid:$cid}), (c:Chunk {chunk_id:$id}) "
                "MERGE (j)-[:HAS_CHUNK]->(c)", {"cid": cid, "id": ch["chunk_id"]})

    def finalise(self) -> None:
        """Denormalised counters that make ranking cheap at query time."""
        self.conn.execute(
            "MATCH (p:Judge)<-[:DECIDED_BY]-(j:Judgment) WITH p, count(j) AS n "
            "SET p.n_judgments = n")

    # ------------------------------------------------------------------ queries
    def rows(self, cypher: str, params: dict | None = None) -> list[dict]:
        res = self.conn.execute(cypher, params or {})
        cols = res.get_column_names()
        return [dict(zip(cols, row)) for row in res]

    def stats(self) -> dict:
        out: dict[str, int] = {}
        for table in ("Judgment", "Judge", "Court", "Act", "Provision", "Area",
                      "Doctrine", "Chunk"):
            out[table] = self.rows(f"MATCH (n:{table}) RETURN count(n) AS c")[0]["c"]
        for rel in ("CITES", "INTERPRETS", "DECIDED_BY", "ABOUT", "INVOKES",
                    "UNDER_ACT", "HAS_CHUNK", "PART_OF"):
            out[rel] = self.rows(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")[0]["c"]
        out["Judgment_with_text"] = self.rows(
            "MATCH (j:Judgment) WHERE j.has_text RETURN count(j) AS c")[0]["c"]
        return out

    def neighbourhood(self, cids: Sequence[str], hops: int = 1,
                      limit: int = 40) -> list[dict]:
        """Precedents around a set of judgments -- the graph half of GraphRAG."""
        if not cids:
            return []
        return self.rows(
            f"""MATCH (s:Judgment)-[c:CITES*1..{max(1, min(hops, 3))}]->(n:Judgment)
                WHERE list_contains($cids, s.cid) AND n.has_text
                RETURN DISTINCT n.cid AS cid, n.title AS title, n.citation AS citation,
                       n.year AS year
                LIMIT $limit""",
            {"cids": list(cids), "limit": limit})

    def citing(self, cids: Sequence[str], limit: int = 20) -> list[dict]:
        """Later judgments that cite these -- catches "is this still good law?"."""
        if not cids:
            return []
        return self.rows(
            """MATCH (later:Judgment)-[c:CITES]->(s:Judgment)
               WHERE list_contains($cids, s.cid)
               RETURN DISTINCT later.cid AS cid, later.title AS title,
                      later.year AS year, c.treatment AS treatment, s.cid AS target
               ORDER BY year DESC LIMIT $limit""",
            {"cids": list(cids), "limit": limit})

    def by_provision(self, pid: str, limit: int = 20) -> list[dict]:
        return self.rows(
            """MATCH (j:Judgment)-[r:INTERPRETS]->(p:Provision {pid: $pid})
               WHERE j.has_text
               RETURN j.cid AS cid, j.title AS title, j.year AS year,
                      r.mentions AS mentions
               ORDER BY r.mentions DESC, j.year DESC LIMIT $limit""",
            {"pid": pid, "limit": limit})

    def most_cited(self, limit: int = 20, area: str | None = None) -> list[dict]:
        if area:
            return self.rows(
                """MATCH (j:Judgment)-[:ABOUT]->(a:Area {key: $area})
                   OPTIONAL MATCH (j)<-[c:CITES]-()
                   RETURN j.cid AS cid, j.title AS title, j.year AS year,
                          count(c) AS in_degree
                   ORDER BY in_degree DESC LIMIT $limit""",
                {"area": area, "limit": limit})
        return self.rows(
            """MATCH (j:Judgment)<-[c:CITES]-()
               RETURN j.cid AS cid, j.title AS title, j.citation AS citation,
                      j.year AS year, count(c) AS in_degree
               ORDER BY in_degree DESC LIMIT $limit""",
            {"limit": limit})

    def area_coverage(self) -> list[dict]:
        return self.rows(
            """MATCH (a:Area)<-[:ABOUT]-(j:Judgment)
               RETURN a.key AS area, a.name AS name, count(j) AS n
               ORDER BY n DESC""")

    def chunks_for(self, cids: Sequence[str]) -> list[dict]:
        if not cids:
            return []
        return self.rows(
            """MATCH (j:Judgment)-[:HAS_CHUNK]->(c:Chunk)
               WHERE list_contains($cids, j.cid)
               RETURN c.chunk_id AS chunk_id, c.vector_row AS row, j.cid AS cid
               ORDER BY c.ordinal""", {"cids": list(cids)})

    def judgments(self, cids: Sequence[str]) -> dict[str, dict]:
        if not cids:
            return {}
        rows = self.rows(
            """MATCH (j:Judgment) WHERE list_contains($cids, j.cid)
               RETURN j.cid AS cid, j.title AS title, j.citation AS citation,
                      j.year AS year, j.court AS court, j.disposal AS disposal""",
            {"cids": list(cids)})
        return {r["cid"]: r for r in rows}

    # ------------------------------------------------------------------ export
    def export_neo4j_csv(self, out_dir: str | Path) -> Path:
        """Node/edge CSVs plus a LOAD CSV script, for anyone who wants the Neo4j browser."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        import csv

        specs = {
            "judgments": ("MATCH (j:Judgment) RETURN j.cid AS cid, j.title AS title, "
                          "j.citation AS citation, j.year AS year, j.court AS court, "
                          "j.disposal AS disposal, j.has_text AS has_text"),
            "judges": "MATCH (p:Judge) RETURN p.name AS name, p.n_judgments AS n_judgments",
            "provisions": ("MATCH (p:Provision) RETURN p.pid AS pid, p.act_key AS act_key, "
                           "p.label AS label"),
            "acts": "MATCH (a:Act) RETURN a.key AS key, a.name AS name, a.category AS category",
            "areas": "MATCH (a:Area) RETURN a.key AS key, a.name AS name, a.parent AS parent",
            "cites": ("MATCH (a:Judgment)-[r:CITES]->(b:Judgment) "
                      "RETURN a.cid AS src, b.cid AS dst, r.treatment AS treatment"),
            "interprets": ("MATCH (j:Judgment)-[r:INTERPRETS]->(p:Provision) "
                           "RETURN j.cid AS src, p.pid AS dst, r.mentions AS mentions"),
            "decided_by": ("MATCH (j:Judgment)-[r:DECIDED_BY]->(p:Judge) "
                           "RETURN j.cid AS src, p.name AS dst, r.authored AS authored"),
            "about": ("MATCH (j:Judgment)-[r:ABOUT]->(a:Area) "
                      "RETURN j.cid AS src, a.key AS dst, r.score AS score"),
        }
        for name, cypher in specs.items():
            rows = self.rows(cypher)
            path = out / f"{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                if rows:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        (out / "load_neo4j.cypher").write_text(
            "// Put these CSVs in your Neo4j import/ directory, then run this file.\n"
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Judgment) REQUIRE j.cid IS UNIQUE;\n"
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Judge) REQUIRE p.name IS UNIQUE;\n"
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Provision) REQUIRE p.pid IS UNIQUE;\n"
            "LOAD CSV WITH HEADERS FROM 'file:///judgments.csv' AS r\n"
            "  CREATE (:Judgment {cid:r.cid, title:r.title, citation:r.citation,\n"
            "                     year:toInteger(r.year), court:r.court});\n"
            "LOAD CSV WITH HEADERS FROM 'file:///cites.csv' AS r\n"
            "  MATCH (a:Judgment {cid:r.src}), (b:Judgment {cid:r.dst})\n"
            "  CREATE (a)-[:CITES {treatment:r.treatment}]->(b);\n",
            encoding="utf-8")
        return out


def build_resolver(metas: Iterable[dict]) -> dict[str, str]:
    """citation key -> cid, so CITES edges land on real nodes where we have them."""
    resolver: dict[str, str] = {}
    for m in metas:
        for field in ("citation", "neutral_citation"):
            for key in citation_keys(m.get(field)):
                resolver.setdefault(key, m["cid"])
    return resolver
