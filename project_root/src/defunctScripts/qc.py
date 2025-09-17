#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import json
import cv2
import pandas as pd
import yaml
from src.common.io import ensure_dir, write_jsonl, now_iso


def load_manifest(out_dir: Path, stem: str) -> dict:
    return json.loads((out_dir / "meta" / f"{stem}.ingest.json").read_text())


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.qc <config.yaml> <video_path>", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    video_path = Path(sys.argv[2])
    paths = cfg["paths"]
    out_dir = Path(paths["out"])
    logs_dir = Path(paths["logs"])
    stem = video_path.stem

    tracks = pd.read_parquet(out_dir / "tables" / f"{stem}_tracks.parquet")
    fps = float(load_manifest(out_dir, stem)["fps"])
    stride = int(cfg["qc"].get("overlay_stride", 5))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        write_jsonl(
            Path(logs_dir) / "qc.jsonl",
            [
                {
                    "stage": "qc",
                    "video": str(video_path),
                    "t": now_iso(),
                    "level": "ERROR",
                    "msg": "cannot_open",
                }
            ],
        )
        sys.exit(1)

    overlay_dir = out_dir / "qc" / "overlay" / stem
    ensure_dir(overlay_dir)
    out_mp4 = str(out_dir / "qc" / f"{stem}_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_mp4, fourcc, max(fps, 1.0), (w, h))

    # build quick frame index for faster loc
    tracks = tracks.sort_values(["frame", "roi_id"])
    grp = tracks.groupby(["frame", "roi_id"], as_index=False).first()

    frame_idx = 0
    ok_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            # draw any points for this frame
            rows = grp[grp["frame"] == frame_idx]
            for _, r in rows.iterrows():
                if pd.notna(r["x"]) and pd.notna(r["y"]):
                    cv2.circle(frame, (int(r["x"]), int(r["y"])), 4, (0, 255, 0), -1)
            vw.write(frame)
            ok_count += 1
            if ok_count % 100 == 0:
                cv2.imwrite(str(overlay_dir / f"overlay_{frame_idx:06d}.png"), frame)

        frame_idx += 1

    cap.release()
    vw.release()
    write_jsonl(
        Path(logs_dir) / "qc.jsonl",
        [
            {
                "stage": "qc",
                "video": str(video_path),
                "t": now_iso(),
                "level": "INFO",
                "msg": "ok",
                "frames_written": ok_count,
                "mp4": out_mp4,
            }
        ],
    )


if __name__ == "__main__":
    main()
