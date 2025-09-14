#!/usr/bin/env python3
from __future__ import annotations
import json
import pathlib
import sys
import yaml
import collections


def read_jsonl(path: pathlib.Path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.diag <config.yaml>", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(open(sys.argv[1]))
    logs = pathlib.Path(cfg["paths"]["logs"])
    out = pathlib.Path(cfg["paths"]["out"])

    issues = []

    def scan(logname):
        rows = read_jsonl(logs / logname)
        level_counts = collections.Counter(r.get("level", "INFO") for r in rows)
        return rows, level_counts

    rows, counts = scan("ingest.jsonl")
    print(f"[ingest] log entries: {len(rows)}  levels: {dict(counts)}")
    for r in rows:
        if r.get("level") == "ERROR":
            issues.append(("ingest", r.get("msg"), r.get("video")))

    metas = list((out / "meta").glob("*.ingest.json"))
    print(f"[ingest] manifests: {len(metas)}")
    for m in metas:
        j = json.loads(m.read_text())
        missing = [
            k
            for k in ("fps", "frames", "width", "height", "sha256")
            if j.get(k) in (None, 0, "", float("nan"))
        ]
        if missing:
            issues.append(("ingest", f"missing fields: {missing}", m.name))

    if issues:
        print("\n== Issues ==")
        for s, msg, ref in issues:
            print(f"- {s}: {msg} ({ref})")
        sys.exit(1)

    print("\nAll good 👍")


if __name__ == "__main__":
    main()
