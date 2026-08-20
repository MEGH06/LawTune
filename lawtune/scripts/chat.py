#!/usr/bin/env python
"""Interactive LawTune chat with the full guardrail stack in the loop.

Every turn goes: input screen -> generate -> citation validation -> disclaimer. The
prompt shape is byte-identical to what 03_build_datasets.py trained on; if you change
SYSTEM_PROMPT in config.py you must retrain, not just restart.

    python scripts/chat.py
    python scripts/chat.py --rag                    # ground answers in the graph + FAISS
    python scripts/chat.py --model outputs/lawtune_adapter --temperature 0.2
    python scripts/chat.py --no-guardrails          # see what the raw model does

With --rag every question is first answered by retrieval: FAISS finds passages, the
knowledge graph pulls the precedents around them and flags anything later overruled, and
the model is told to answer from that and say so when it cannot. That is the only
configuration in which a 2B model should be trusted with a section number.

Commands: /reset  /flags  /lang <code>  /raw  /sources  /quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawtune.config import LANGUAGES, MAX_SEQ_LEN, PROCESSED  # noqa: E402
from lawtune.guardrails import check_input, check_output  # noqa: E402
from lawtune.model import LawTuneModel  # noqa: E402

GRAPH = PROCESSED / "legal_graph.kuzu"
INDEX = PROCESSED / "faiss_index"


class LawTune:
    def __init__(self, model_path: str = "", seq_len: int = MAX_SEQ_LEN,
                 temperature: float = 0.3, use_guardrails: bool = True,
                 rag=None):
        self.bot = LawTuneModel(model_path, seq_len)
        self.temperature = temperature
        self.use_guardrails = use_guardrails
        self.rag = rag
        self.history: list[dict] = []
        self.last_flags: list[str] = []
        self.last_raw: str = ""
        self.last_retrieval = None
        self.expected_script: str | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_flags.clear()

    def ask(self, question: str, max_new_tokens: int = 512, stream: bool = True) -> str:
        if self.use_guardrails:
            verdict = check_input(question)
            self.last_flags = list(verdict.flags)
            if not verdict.allowed:
                print(verdict.replacement)
                self._remember(question, verdict.replacement or "")
                return verdict.replacement or ""

        prompt = question
        if self.rag is not None:
            prompt, retrieval = self.rag.answer_prompt(question)
            self.last_retrieval = retrieval
            n = len([c for c in retrieval.candidates if c.passages])
            warn = sum(len(c.warnings) for c in retrieval.candidates)
            suffix = f", {warn} later-treatment warnings" if warn else ""
            print(f"[retrieved {n} judgments{suffix}]\n")
            # History is dropped for RAG turns: the retrieved context is already long,
            # and a 2B model given both starts blending prior answers into new ones.
            raw = self.bot.ask(prompt, max_new_tokens=max_new_tokens,
                               temperature=self.temperature, stream=stream)
        else:
            raw = self.bot.ask(question, history=self.history,
                               max_new_tokens=max_new_tokens,
                               temperature=self.temperature, stream=stream)
        self.last_raw = raw
        if not stream:
            print(raw)

        if not self.use_guardrails:
            self._remember(question, raw)
            return raw

        verdict = check_output(raw, question, expected_script=self.expected_script)
        self.last_flags += verdict.flags
        final = verdict.replacement or raw
        if not verdict.allowed:
            print("\n\n[guardrail] withheld the generated answer:\n")
            print(final)
        elif verdict.flags:
            print(f"\n[guardrail flags: {', '.join(verdict.flags)}]")
        self._remember(question, final)
        return final

    def _remember(self, question: str, answer: str) -> None:
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]
        # Keep the window bounded; a 2B model degrades fast with long histories.
        self.history = self.history[-8:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-guardrails", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--rag", action="store_true",
                    help="ground every answer in the knowledge graph + FAISS index")
    ap.add_argument("--graph", type=str, default=str(GRAPH))
    ap.add_argument("--index", type=str, default=str(INDEX))
    ap.add_argument("--fallback-embeddings", action="store_true")
    args = ap.parse_args()

    rag = None
    if args.rag:
        from lawtune.kg.retrieve import GraphRAG

        for path, hint in ((args.graph, "09_build_graph.py"),
                           (args.index, "10_build_index.py")):
            if not Path(path).exists():
                raise SystemExit(f"{path} not found. Run scripts/{hint} first, "
                                 f"or drop --rag.")
        print("loading knowledge graph + vector index ...")
        rag = GraphRAG(args.graph, args.index,
                       fallback_embeddings=args.fallback_embeddings)

    bot = LawTune(args.model, args.seq_len, args.temperature,
                  not args.no_guardrails, rag=rag)
    print(f"\nLawTune ready  [{bot.bot.model_id}]"
          + ("  [RAG: grounded in retrieved judgments]" if rag else ""))
    print("Ask in any of the 22 scheduled languages or English.")
    print("Commands: /reset  /flags  /lang <code>  /raw  /sources  /quit\n")

    while True:
        try:
            q = input("\n\033[1myou >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            break
        if q == "/reset":
            bot.reset()
            print("history cleared")
            continue
        if q == "/flags":
            print(bot.last_flags or "no flags on the last turn")
            continue
        if q == "/raw":
            print(bot.last_raw or "nothing generated yet")
            continue
        if q == "/sources":
            r = bot.last_retrieval
            if r is None:
                print("no retrieval on the last turn (is --rag on?)")
            else:
                for i, c in enumerate(r.candidates, 1):
                    if not c.passages:
                        continue
                    print(f"  [{i}] {c.title or c.cid}  {c.citation}  "
                          f"(score {c.final_score:.3f})")
                    for w in c.warnings:
                        print(f"       /!\ {w}")
            continue
        if q.startswith("/lang"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in LANGUAGES:
                bot.expected_script = LANGUAGES[parts[1]][3]
                print(f"expecting {LANGUAGES[parts[1]][0]} ({bot.expected_script}) replies")
            else:
                print("codes:", ", ".join(LANGUAGES))
            continue

        print("\n\033[1mlawtune >\033[0m ", end="", flush=True)
        bot.ask(q, args.max_new_tokens, stream=not args.no_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
