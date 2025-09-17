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


def load_manifest_or_probe(
    out_dir: Path, video_path: Path, cap: cv2.VideoCapture
) -> dict:
    """
    Try to load ingest meta; if missing, probe via OpenCV.
    Returns dict with width, height, fps, and stem.
    """
    stem = video_path.stem
    meta_path = out_dir / "meta" / f"{stem}.ingest.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # Ensure essential fields exist
        for k in ("width", "height", "fps"):
            if k not in meta:
                raise KeyError(f"Missing '{k}' in {meta_path}")
        return meta
    # Fallback: probe from cap
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return {"width": width, "height": height, "fps": fps, "stem": stem}


def get_rois(cfg: dict, meta: dict):
    """
    Accept ROIs from either cfg['video']['rois'] or cfg['detect_track']['rois'].
    Return list[(roi_id, mask_uint8, (x,y,w,h))]. If none provided, returns one full-frame ROI.
    """
    w, h = int(meta["width"]), int(meta["height"])

    def _rects(root: dict):
        rects = (root or {}).get("rects", [])
        out = []
        for r in rects:
            x, y = int(r["x"]), int(r["y"])
            ww, hh = int(r["w"]), int(r["h"])
            # clamp to frame
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            ww = max(1, min(ww, w - x))
            hh = max(1, min(hh, h - y))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y : y + hh, x : x + ww] = 255
            out.append((str(r.get("id", f"roi_{len(out)}")), mask, (x, y, ww, hh)))
        return out

    rois = []
    # Prefer video.rois if present
    v_rois = (cfg.get("video") or {}).get("rois") or {}
    if v_rois.get("mode") == "rects":
        rois = _rects(v_rois)

    # Fallback / supplement: detect_track.rois
    if not rois:
        dt_rois = (cfg.get("detect_track") or {}).get("rois") or {}
        if dt_rois.get("mode") == "rects":
            rois = _rects(dt_rois)

    # Default full-frame if none
    if not rois:
        mask = np.full((h, w), 255, np.uint8)
        rois = [("full", mask, (0, 0, w, h))]

    return rois


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.detect_track <config.yaml> <video_path>",
            file=sys.stderr,
        )
        sys.exit(2)

    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    video_path = Path(sys.argv[2])

    paths = cfg.get("paths", {})
    out_dir = Path(paths.get("out") or paths.get("out_dir") or "out")
    logs_dir = Path(paths.get("logs") or paths.get("logs_dir") or "logs")
    ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "meta")
    ensure_dir(logs_dir)

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

    meta = load_manifest_or_probe(out_dir, video_path, cap)
    fps = float(meta.get("fps", 0.0))
    stem = video_path.stem

    # --- detection params ---
    cfg_det = cfg["detect_track"]
    min_area = int(cfg_det.get("min_area", 50))
    max_area = int(cfg_det.get("max_area", 50000))
    morph_k = int(cfg_det.get("morph_kernel", 3))
    max_jump = float(cfg_det.get("max_jump", 0))  # 0 = disable jump filtering

    rois = get_rois(cfg, meta)

    # Background subtractor (you can wire cfg if you want)
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))

    rows = []
    frame_idx = 0
    # Keep a previous centroid per ROI for simple continuity gating
    prev_xy = {roi_id: None for roi_id, _, _ in rois}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for roi_id, mask, (x, y, w, h) in rois:
            # Masked grayscale (same full-frame size, non-ROI pixels are 0)
            g = cv2.bitwise_and(gray, gray, mask=mask)

            # Foreground segmentation
            fg = mog2.apply(g)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

            # Restrict to the ROI box before contouring to avoid border junk
            fg_roi = fg[y : y + h, x : x + w]

            cnts, _ = cv2.findContours(
                fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # choose largest contour in ROI
            best = None
            best_area = 0.0
            cx = cy = None

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
                    # NOTE: cx/cy in ROI coords → convert to full-frame coords
                    cx = float(M["m10"] / M["m00"]) + x
                    cy = float(M["m01"] / M["m00"]) + y

            # Optional jump filter for continuity
            if (
                max_jump > 0
                and prev_xy[roi_id] is not None
                and cx is not None
                and cy is not None
            ):
                px, py = prev_xy[roi_id]
                if (cx - px) ** 2 + (cy - py) ** 2 > max_jump**2:
                    # too big a jump → treat as missing this frame
                    cx, cy, best_area = None, None, 0.0

            # Update prev if we have a valid point
            if cx is not None and cy is not None:
                prev_xy[roi_id] = (cx, cy)

            t = (frame_idx / fps) if fps > 0 else None

            rows.append(
                {
                    "video": str(video_path.name),
                    "frame": frame_idx,
                    "t": t,
                    "roi_id": roi_id,
                    "x": cx,
                    "y": cy,
                    "area": best_area,
                    "fps": fps,
                }
            )

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    write_table(
        df,
        out_dir / "tables" / f"{stem}_tracks",
        cfg.get("outputs", {}).get("csv_mirror", False),
    )

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
                "rows": int(df.shape[0]),
                "fps": fps,
                "rois": [
                    dict(id=rid, box=dict(x=rx, y=ry, w=rw, h=rh))
                    for rid, _, (rx, ry, rw, rh) in rois
                ],
                "min_area": min_area,
                "max_area": max_area,
                "max_jump": max_jump,
            }
        ],
    )


if __name__ == "__main__":
    main()
