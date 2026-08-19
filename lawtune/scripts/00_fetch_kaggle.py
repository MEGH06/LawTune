#!/usr/bin/env python
"""Pull the Kaggle law dataset into data/raw/ and show what the loader will make of it.

If you already have constitution_qa.json / crpc_qa.json / ipc_qa.json, just drop them
in data/raw/ and run this with --inspect-only.

    python scripts/00_fetch_kaggle.py --dataset <owner>/<dataset-slug>
    python scripts/00_fetch_kaggle.py --inspect-only

Credentials: put kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or export
KAGGLE_USERNAME and KAGGLE_KEY. If a kaggle.json is sitting in the working directory,
this script installs it for you.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawtune.config import RAW  # noqa: E402


def ensure_credentials() -> None:
    home_cred = Path.home() / ".kaggle" / "kaggle.json"
    if home_cred.exists() or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return
    local = Path("kaggle.json")
    if local.exists():
        home_cred.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, home_cred)
        home_cred.chmod(0o600)
        print(f"installed credentials -> {home_cred}")
        return
    print("No Kaggle credentials found. Either:\n"
          "  - place kaggle.json at ~/.kaggle/kaggle.json, or\n"
          "  - export KAGGLE_USERNAME / KAGGLE_KEY, or\n"
          "  - skip this script and copy your JSON files into data/raw/ by hand.",
          file=sys.stderr)


def download(slug: str) -> None:
    ensure_credentials()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:                                     # noqa: BLE001
        raise SystemExit(f"pip install kaggle  ({type(exc).__name__}: {exc})") from exc

    api = KaggleApi()
    api.authenticate()
    print(f"downloading {slug} -> {RAW}")
    api.dataset_download_files(slug, path=str(RAW), unzip=True, quiet=False)


def inspect() -> None:
    from lawtune import law_data

    files = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in {".json", ".jsonl"})
    if not files:
        print(f"nothing in {RAW}")
        return
    print(f"\nfiles in {RAW}:")
    for f in files:
        print(f"  {f.relative_to(RAW)}  ({f.stat().st_size / 1e6:.2f} MB)")

    print("\nparsing:")
    records = law_data.load_all()
    if not records:
        return
    with_cite = sum(1 for r in records if r["citation"])
    lens = sorted(len(r["answer"]) for r in records)
    print(f"\n  with a detectable citation : {with_cite} ({100 * with_cite / len(records):.0f}%)")
    print(f"  answer length p50/p90/max  : {lens[len(lens) // 2]} / "
          f"{lens[int(len(lens) * 0.9)]} / {lens[-1]} chars")
    print("\nsample record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:700])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="",
                    help="Kaggle slug, e.g. someone/indian-law-qa")
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    if args.dataset and not args.inspect_only:
        download(args.dataset)
    inspect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
