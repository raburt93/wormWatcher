#!/usr/bin/env python3
"""
Ingest: read video, extract metadata, hash, and snapshot frames.
Writes:
  logs/ingest.jsonl                  # per-run log
  out/meta/<video_stem>.ingest.json  # metadata manifest
  out/qc/ingest/<stem>/frame_XXXXX.png  # snapshots
"""

from __future__ import annotations
import os
import sys
import hashlib
import pathlib
import cv2
import pandas as pd
import yaml
from src.common.io import write_json, write_jsonl, now_iso, ensure_dir


def sha256(path: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingest <config.yaml> <video_path>", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(open(sys.argv[1]))
    video_path = pathlib.Path(sys.argv[2])
    paths = cfg["paths"]
    out_dir = pathlib.Path(paths["out"])
    logs_dir = pathlib.Path(paths["logs"])
    csv_mirror = cfg["outputs"]["csv_mirror"]
    stride = int(cfg["outputs"]["snapshot_stride"])

    cap = cv2.VideoCapture(str(video_path))
    ok = cap.isOpened()
    meta = {
        "stage": "ingest",
        "video": str(video_path),
        "exists": video_path.exists(),
        "opened": ok,
        "t_start": now_iso(),
        "config": {"snapshot_stride": stride},
    }
    if not ok:
        write_jsonl(
            logs_dir / "ingest.jsonl", [dict(meta, level="ERROR", msg="cannot_open")]
        )
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or float("nan")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = nframes / fps if fps and fps > 0 else None

    hv = sha256(video_path)
    manifest = {
        "video_stem": video_path.stem,
        "path": str(video_path),
        "sha256": hv,
        "fps": fps,
        "frames": nframes,
        "width": w,
        "height": h,
        "duration_s": duration,
        "source": {"hostname": os.uname().nodename},
    }

    snap_dir = out_dir / "qc" / "ingest" / video_path.stem
    ensure_dir(snap_dir)
    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            cv2.imwrite(str(snap_dir / f"frame_{i:06d}.png"), frame)
            saved += 1
        i += 1
    cap.release()

    write_json(out_dir / "meta" / f"{video_path.stem}.ingest.json", manifest)
    write_jsonl(
        logs_dir / "ingest.jsonl",
        [
            dict(
                meta,
                level="INFO",
                msg="ok",
                fps=fps,
                frames=nframes,
                width=w,
                height=h,
                duration_s=duration,
                sha256=hv,
                snapshots=saved,
                t_end=now_iso(),
            )
        ],
    )

    # Tiny summary table
    df = pd.DataFrame([manifest])
    from src.common.io import write_table

    write_table(
        df, out_dir / "tables" / f"{video_path.stem}_ingest_summary", csv_mirror
    )


if __name__ == "__main__":
    main()
