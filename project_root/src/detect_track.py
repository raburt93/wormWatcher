#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import yaml
from src.common.io import ensure_dir, write_jsonl, now_iso, write_table


def load_manifest(out_dir: Path, stem: str) -> dict:
    p = out_dir / "meta" / f"{stem}.ingest.json"
    return json.loads(p.read_text())


def build_roi_masks(meta: dict, cfg_detect: dict):
    w, h = int(meta["width"]), int(meta["height"])
    rects = cfg_detect.get("rois", {}).get("rects", [])
    masks = []
    if rects:
        for r in rects:
            mask = np.zeros((h, w), dtype=np.uint8)
            x, y, ww, hh = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            mask[y : y + hh, x : x + ww] = 255
            masks.append((r["id"], mask, (x, y, ww, hh)))
    else:
        masks.append(("full", np.full((h, w), 255, np.uint8), (0, 0, w, h)))
    return masks


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.detect_track <config.yaml> <video_path>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    video_path = Path(sys.argv[2])
    paths = cfg["paths"]
    out_dir = Path(paths["out"])
    logs_dir = Path(paths["logs"])
    ensure_dir(out_dir / "tables")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        write_jsonl(
            Path(logs_dir) / "detect_track.jsonl",
            [
                {
                    "stage": "detect_track",
                    "video": str(video_path),
                    "t": now_iso(),
                    "level": "ERROR",
                    "msg": "cannot_open",
                }
            ],
        )
        sys.exit(1)

    stem = video_path.stem
    meta = load_manifest(out_dir, stem)
    fps = float(meta["fps"])
    cfg_det = cfg["detect_track"]
    min_area = int(cfg_det["min_area"])
    max_area = int(cfg_det["max_area"])
    morph_k = int(cfg_det.get("morph_kernel", 3))

    rois = build_roi_masks(meta, cfg_det)
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))

    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for roi_id, mask, (x, y, w, h) in rois:
            g = cv2.bitwise_and(gray, gray, mask=mask)
            fg = mog2.apply(g)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # choose largest contour in ROI
            best = None
            best_area = 0
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area < min_area or area > max_area:
                    continue
                if area > best_area:
                    best = c
                    best_area = area

            if best is not None:
                M = cv2.moments(best)
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                else:
                    cx, cy = None, None
            else:
                cx, cy, best_area = None, None, 0.0

            t = frame_idx / fps if fps > 0 else None
            rows.append(
                {
                    "frame": frame_idx,
                    "t": t,
                    "roi_id": roi_id,
                    "x": cx,
                    "y": cy,
                    "area": best_area,
                }
            )

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "tables" / f"{stem}_tracks", cfg["outputs"]["csv_mirror"])
    write_jsonl(
        Path(logs_dir) / "detect_track.jsonl",
        [
            {
                "stage": "detect_track",
                "video": str(video_path),
                "t": now_iso(),
                "level": "INFO",
                "msg": "ok",
                "frames": frame_idx,
                "rows": len(rows),
            }
        ],
    )


if __name__ == "__main__":
    main()
