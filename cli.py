from __future__ import annotations

import argparse
import json

from app.indexer import ingest
from app.rag import answer_question


def main() -> None:
    p = argparse.ArgumentParser(description="Alrouf RAG KB MVP (Light)")
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--question", type=str)
    args = p.parse_args()

    if args.ingest:
        print(json.dumps({"ok": True, "stats": ingest(rebuild=args.rebuild)}, ensure_ascii=False, indent=2))

    if args.question:
        print(json.dumps(answer_question(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
