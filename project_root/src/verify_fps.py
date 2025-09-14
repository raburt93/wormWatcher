#!/usr/bin/env python3
"""
Verify video facts vs ingest manifest and check FPS constancy.

Usage:
  python -m src.verify_fps config.yaml videos/YourFile.m4v [videos/MoreFiles...]

Outputs (per file):
- SHA256 match vs manifest
- FPS(manifest), FPS(estimate: median Δt, mean Δt, and slope over span)
- max_jitter (s), unique Δt bins with counts
- Verdict: CFR-like (binning-only jitter) vs VFR (broad jitter)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics as stats
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import yaml

TOL_FPS = 1e-3  # ±0.001 fps
TOL_DT = 5e-4  # 0.0005 s (0.5 ms) jitter window
MAX_FRAMES_CHECK = 2000


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_manifest(out_dir: Path, stem: str) -> Optional[dict]:
    p = out_dir / "meta" / f"{stem}.ingest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _run_ffprobe_json(cmd: list[str]) -> Optional[dict]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def ffprobe_frame_times(
    video: str, max_frames: int = MAX_FRAMES_CHECK
) -> Optional[List[float]]:
    """Return a list of frame timestamps (seconds) using ffprobe JSON."""
    if not shutil.which("ffprobe"):
        return None
    base_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time",
        "-of",
        "json",
    ]
    data = _run_ffprobe_json(
        base_cmd + ["-read_intervals", f"0%+#{max_frames}", video]
    ) or _run_ffprobe_json(base_cmd + [video])
    if not data:
        return None

    frames = data.get("frames", [])
    times: List[float] = []
    for fr in frames[:max_frames]:
        t = fr.get("best_effort_timestamp_time") or fr.get("pkt_pts_time")
        if not t or t == "N/A":
            continue
        try:
            times.append(float(t))
        except ValueError:
            continue
    return times or None


def describe_dts(times: List[float]) -> dict:
    dts = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    if not dts:
        return {
            "n": 0,
            "med_dt": float("nan"),
            "mean_dt": float("nan"),
            "max_jit": float("nan"),
            "bins": {},
            "fps_med": float("nan"),
            "fps_mean": float("nan"),
            "fps_slope": float("nan"),
        }

    med_dt = stats.median(dts)
    mean_dt = sum(dts) / len(dts)
    max_jit = max(abs(dt - med_dt) for dt in dts)

    span = times[-1] - times[0]
    fps_slope = (len(times) - 1) / span if span > 0 else float("nan")

    def round_us(x: float) -> float:
        return round(x, 6)

    bins = Counter(round_us(dt) for dt in dts)

    def sort_key(kv: tuple[float, int]) -> tuple[int, float]:
        return (-kv[1], kv[0])

    bins = dict(sorted(bins.items(), key=sort_key))

    return {
        "n": len(dts),
        "med_dt": med_dt,
        "mean_dt": mean_dt,
        "max_jit": max_jit,
        "bins": bins,
        "fps_med": (1.0 / med_dt) if med_dt > 0 else float("nan"),
        "fps_mean": (1.0 / mean_dt) if mean_dt > 0 else float("nan"),
        "fps_slope": fps_slope,
    }


def analyze_one(cfg: dict, video_path: Path) -> int:
    out_dir = Path(cfg["paths"]["out"])
    stem = video_path.stem
    manifest = read_manifest(out_dir, stem)
    if not manifest:
        print(
            f"[{video_path.name}] ERR: ingest manifest not found. Run `make ingest VIDEO={video_path}` first."
        )
        return 2

    want_sha = manifest["sha256"]
    want_fps = float(manifest["fps"])
    want_frames = int(manifest["frames"])

    got_sha = sha256(video_path)
    ok_sha = got_sha == want_sha
    print(f"[{video_path.name}] sha256: {got_sha}  (matches manifest: {ok_sha})")

    times = ffprobe_frame_times(str(video_path), MAX_FRAMES_CHECK)
    if times:
        desc = describe_dts(times)
        print(
            f"[{video_path.name}] fps(manifest)={want_fps:.6f}  "
            f"fps(med)={desc['fps_med']:.6f}  fps(mean)={desc['fps_mean']:.6f}  fps(slope)={desc['fps_slope']:.6f}"
        )
        print(
            f"[{video_path.name}] max_jitter={desc['max_jit']:.6f}s  Δt bins (s, count): {desc['bins']}"
        )
        nominal = 1.0 / want_fps if want_fps > 0 else float("nan")
        bins_ok = all(abs(dt - nominal) <= 0.0015 for dt in desc["bins"].keys())
        verdict = "CFR-like (timebase quantization)" if bins_ok else "VFR"
        tol_ok = abs(desc["fps_slope"] - want_fps) <= 0.01
        print(
            f"[{video_path.name}] Const-FPS verdict: {verdict}  (window-slope match: {'OK' if tol_ok else 'NOT OK'})"
        )
    else:
        print(
            f"[{video_path.name}] ffprobe not available; relying on OpenCV manifest only."
        )

    print(f"[{video_path.name}] Frames(manifest): {want_frames}")
    print(f"[{video_path.name}] FPS(manifest):    {want_fps:.6f}")
    print(f"[{video_path.name}] Hash match:       {'YES' if ok_sha else 'NO'}\n")
    return 0 if ok_sha else 1


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(
            "Usage: python -m src.verify_fps <config.yaml> <video_path> [more_videos...]",
            file=sys.stderr,
        )
        return 2
    cfg = yaml.safe_load(Path(argv[1]).read_text())
    exit_codes: List[int] = []
    for arg in argv[2:]:
        video = Path(arg)
        if not video.exists():
            print(f"[{video.name}] ERR: file not found")
            exit_codes.append(2)
            continue
        exit_codes.append(analyze_one(cfg, video))
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
