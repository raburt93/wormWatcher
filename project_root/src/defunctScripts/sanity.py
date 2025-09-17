import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def frame_hash(frame: np.ndarray) -> str:
    """Return short SHA256 hash of a frame's bytes (first 16 hex chars)."""
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]


def main(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(json.dumps({"ok": False, "error": "cannot_open"}))
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample = min(300, total_frames)
    hashes: list[str] = []
    motion: list[float] = []
    prev = None

    for _ in range(sample):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hashes.append(frame_hash(gray))
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            motion.append(float(np.mean(diff)))
        prev = gray

    cap.release()

    out = {
        "ok": True,
        "fps": float(fps),
        "frames_total": total_frames,
        "w": width,
        "h": height,
        "sampled": len(hashes),
        "motion_energy_mean": float(np.mean(motion)) if motion else None,
        "first5_hashes": hashes[:5],
        "input_path": path,
    }

    log_dir = Path("project_root/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "ingest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "no_video_arg"}))
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
