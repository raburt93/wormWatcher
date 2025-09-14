import cv2, sys, json, hashlib, numpy as np, os, time
from pathlib import Path

def frame_hash(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]

def main(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(json.dumps({"ok": False, "error": "cannot_open"})); return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample = min(300, n)
    hashes, motion = [], []
    prev = None
    for i in range(sample):
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hashes.append(frame_hash(gray))
        if prev is not None:
            motion.append(float(np.mean(cv2.absdiff(gray, prev))))
        prev = gray
    cap.release()
    out = {
        "ok": True, "fps": fps, "frames_total": n, "w": w, "h": h,
        "sampled": len(hashes),
        "motion_energy_mean": float(np.mean(motion)) if motion else None,
        "first5_hashes": hashes[:5]
    }
    Path("project_root/logs").mkdir(parents=True, exist_ok=True)
    with open("project_root/logs/ingest.jsonl","a") as f: f.write(json.dumps(out)+"\n")
    print(json.dumps(out))
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print(json.dumps({"ok": False, "error": "no_video_arg"})); sys.exit(2)
    sys.exit(main(sys.argv[1]))
