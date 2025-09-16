#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import json
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.common.io import ensure_dir, write_json, write_jsonl, now_iso


def load_manifest(out_dir: Path, stem: str) -> dict:
    p = out_dir / "meta" / f"{stem}.ingest.json"
    return json.loads(p.read_text())


def gaussian(img, ksize: int) -> np.ndarray:
    if ksize <= 1 or ksize % 2 == 0:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.preprocess <config.yaml> <video_path>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    video_path = Path(sys.argv[2])
    paths = cfg["paths"]
    out_dir = Path(paths["out"])
    logs_dir = Path(paths["logs"])
    ensure_dir(out_dir / "qc")
    ensure_dir(out_dir / "figs")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        write_jsonl(
            Path(logs_dir) / "preprocess.jsonl",
            [
                {
                    "stage": "preprocess",
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
    rois = cfg["detect_track"].get("rois", {}).get("rects", [])
    # build ROI masks
    w, h = int(meta["width"]), int(meta["height"])
    roi_masks = []
    if rois:
        for r in rois:
            mask = np.zeros((h, w), dtype=np.uint8)
            x, y, ww, hh = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            mask[y : y + hh, x : x + ww] = 255
            roi_masks.append((r["id"], mask))
    else:
        roi_masks.append(("full", np.full((h, w), 255, np.uint8)))

    # background model
    hist = int(cfg["preprocess"]["bg_history"])
    var_thr = float(cfg["preprocess"]["bg_var_threshold"])
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=hist, varThreshold=var_thr, detectShadows=False
    )

    denoise = cfg["preprocess"]["denoise"]
    gk = int(cfg["preprocess"]["gaussian_ksize"])

    motion_energy = []
    frame_idx = 0
    preview_dir = out_dir / "qc" / "preprocess" / stem
    ensure_dir(preview_dir)
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if denoise == "gaussian":
            gray = gaussian(gray, gk)

        # apply ROI(s)
        fg_sum = 0
        for _, mask in roi_masks:
            g = cv2.bitwise_and(gray, gray, mask=mask)
            fg = mog2.apply(g)
            # light morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            fg_sum += int((fg > 0).sum())

        motion_energy.append(fg_sum)

        if frame_idx % 1000 == 0:
            # save quick preview binary mask
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for _, mask in roi_masks:
                color[mask == 0] = (32, 32, 32)
            cv2.imwrite(str(preview_dir / f"preview_{frame_idx:06d}.png"), color)
            saved += 1
        frame_idx += 1

    cap.release()

    # motion energy plot
    if motion_energy:
        plt.figure()
        plt.plot(motion_energy)
        plt.xlabel("Frame")
        plt.ylabel("Motion energy (fg px)")
        plt.tight_layout()
        fig_path = out_dir / "figs" / f"{stem}_motion_energy.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    # write a small manifest
    out_meta = {
        "video": str(video_path),
        "fps": fps,
        "frames_seen": frame_idx,
        "roi_ids": [rid for rid, _ in roi_masks],
        "bg_model": "mog2",
        "previews_saved": saved,
        "motion_energy_png": str(fig_path),
    }
    ensure_dir(out_dir / "meta")
    write_json(out_dir / "meta" / f"{stem}.preprocess.json", out_meta)
    write_jsonl(
        Path(logs_dir) / "preprocess.jsonl",
        [
            {
                "stage": "preprocess",
                "video": str(video_path),
                "t": now_iso(),
                "level": "INFO",
                "msg": "ok",
                "frames_seen": frame_idx,
                "previews": saved,
            }
        ],
    )


if __name__ == "__main__":
    main()
