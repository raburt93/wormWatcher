#!/usr/bin/env python3
"""
Audio-driven trial splitter for WormWatcher.

Detects 50 trials after acclimation using audio (5–80 Hz band-pass + envelope),
validates light flash in the last 2 s of each 6 s tone via ROI brightness surge,
and makes frame-accurate 8 s clips (1 s lead + 6 s tone + 1 s tail).

Outputs:
  out/tables/<stem>_trials.csv
  out/tables/<stem>_split_qc.csv
  out/figs/<stem>_audio_envelope.png
  out/meta/<stem>.split.json
  out/clips/<stem>/trial_###_<label>.mp4
  logs/split_audio.jsonl

Usage:
  python -m src.split_audio config.yaml videos/D5W1P28Jun25.m4v \
      --min-start-s 600 --expected 50 --trial-dur 6 --lead 1 --tail 1 --reencode
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.signal import butter, filtfilt, hilbert

import csv
import json
import shutil
import subprocess
import sys
import cv2
import numpy as np
import soundfile as sf
import yaml

# ---------- helpers & IO ----------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def write_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def read_ingest(out_dir: Path, stem: str) -> dict:
    p = out_dir / "meta" / f"{stem}.ingest.json"
    return json.loads(p.read_text())


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio_wav(src: Path, dst_wav: Path, sr: int = 2000) -> None:
    """Extract mono WAV at sr Hz via ffmpeg (robust for .m4v/.mp4)."""
    ensure_dir(dst_wav.parent)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        "-y",
        str(dst_wav),
    ]
    subprocess.run(cmd, check=True)


def build_roi_mask(meta: dict, cfg: dict) -> np.ndarray:
    w, h = int(meta["width"]), int(meta["height"])
    rects = cfg.get("detect_track", {}).get("rois", {}).get("rects", [])
    mask = np.zeros((h, w), dtype=np.uint8)
    if rects:
        for r in rects:
            x, y, ww, hh = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            mask[y : y + hh, x : x + ww] = 255
    else:
        mask[:] = 255
    return mask


def export_roi_mask_png(meta: dict, cfg: dict, out_dir: Path, stem: str) -> Path:
    """Save the current ROI mask as a PNG for provenance; return its path."""
    mask = build_roi_mask(meta, cfg)
    roi_png = out_dir / "qc" / f"{stem}_roi_mask.png"
    ensure_dir(roi_png.parent)
    # Save binary mask (0/255)
    import imageio.v2 as iio

    iio.imwrite(roi_png, mask)
    return roi_png


def detect_flashes(
    video: Path,
    meta: dict,
    cfg: dict,
    min_start_s: float,
    sample_hz: float = 10.0,
    smooth_win: int = 9,
    q_lo: float = 0.90,
    q_hi: float = 0.995,
) -> List[float]:
    """Detect bright flashes in ROI as sharp luminance surges; return their start times (s)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    mask = build_roi_mask(meta, cfg)
    fps = float(meta["fps"])
    stride = max(1, int(round(fps / sample_hz)))
    t_list: List[float] = []
    y_list: List[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            t = idx / fps
            if t >= min_start_s:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = cv2.bitwise_and(gray, gray, mask=mask)
                y = cv2.mean(roi, mask=mask)[0]
                t_list.append(t)
                y_list.append(float(y))
        idx += 1
    cap.release()
    if not y_list:
        return []

    y = np.array(y_list, dtype=np.float32)
    # smooth
    if smooth_win > 1 and smooth_win % 2 == 1:
        pad = smooth_win // 2
        y = np.convolve(
            np.pad(y, (pad, pad), mode="edge"),
            np.ones(smooth_win, np.float32) / smooth_win,
            mode="valid",
        )
    # adaptive threshold search to get ~45 flashes (9 per block × 5 cycles)
    target_flashes = 45
    best: Tuple[int, float, np.ndarray] = (0, 0.95, np.array([], dtype=int))
    q = 0.95
    for _ in range(14):
        thr = float(np.quantile(y, q))
        peaks = (y >= thr).astype(np.uint8)
        # collapse consecutive ones into single peaks at rising edges
        rise = np.where((peaks[1:] == 1) & (peaks[:-1] == 0))[0] + 1
        cnt = int(rise.size)
        if not best[2].size or abs(cnt - target_flashes) < abs(
            best[0] - target_flashes
        ):
            best = (cnt, q, rise)
        if cnt == target_flashes:
            break
        if cnt > target_flashes:
            q = min(q_hi, q + 0.01)
        else:
            q = max(q_lo, q - 0.01)
    rise = best[2]
    flashes = [float(t_list[i]) for i in rise.tolist()]
    return flashes


# ---------- audio detection ----------


@dataclass
class Detection:
    onset_s: float
    dur_s: float


def bandpass_envelope(
    x: np.ndarray, sr: int, lo: float = 5.0, hi: float = 80.0, smooth_sec: float = 0.2
) -> np.ndarray:
    """Band-pass and envelope (Hilbert) with moving-average smoothing."""
    nyq = sr / 2.0
    lo_n = max(0.001, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    b, a = butter(4, [lo_n, hi_n], btype="bandpass")
    y = filtfilt(b, a, x)
    env = np.abs(hilbert(y))
    win = max(1, int(round(sr * smooth_sec)))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    padded = np.pad(env, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, k, mode="valid")
    return smoothed.astype(np.float32)


def segments_above_threshold(
    env: np.ndarray, sr: int, thr: float, min_len: float, max_len: float
) -> List[Tuple[int, int]]:
    """Return contiguous index spans where env>=thr and length within [min_len, max_len]."""
    above = env >= thr
    spans: List[Tuple[int, int]] = []
    n = len(env)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i + 1
        while j < n and above[j]:
            j += 1
        length_s = (j - i) / sr
        if length_s >= min_len and length_s <= max_len:
            spans.append((i, j))
        i = j
    return spans


def _segments_above_quantiles(
    seg: np.ndarray, sr: int, q_list: List[float], min_len: float, max_len: float
) -> Optional[int]:
    for q in q_list:
        thr = float(np.quantile(seg, q))
        spans = segments_above_threshold(seg, sr, thr, min_len, max_len)
        if spans:
            return spans[0][0]
    return None


def _segments_above_stats(
    seg: np.ndarray, sr: int, k_list: List[float], min_len: float, max_len: float
) -> Optional[int]:
    mu = float(seg.mean())
    sd = float(seg.std() or 1.0)
    for k in k_list:
        thr = mu + k * sd
        spans = segments_above_threshold(seg, sr, thr, min_len, max_len)
        if spans:
            return spans[0][0]
    return None


def _matched_filter_rect(seg: np.ndarray, sr: int, dur_s: float) -> Optional[int]:
    n = max(1, int(round(dur_s * sr)))
    if len(seg) < n:
        return None
    # z-score then correlate with a flat “rectangular” kernel
    z = (seg - seg.mean()) / (seg.std() or 1.0)
    k = np.ones(n, dtype=np.float32) / float(n)
    corr = np.convolve(z, k, mode="valid")
    peak = int(np.argmax(corr))
    # Return start index of best rectangle
    return peak


def infer_sound_only_from_audio(
    env: np.ndarray,
    sr: int,
    between: Tuple[float, float],
    min_len: float = 5.3,
    max_len: float = 7.5,
) -> Optional[float]:
    """
    Search env between (t_lo, t_hi) for one ~6 s plateau; return onset or None.
    Tries (a) quantile thresholds, (b) mean+k·std, (c) matched filter, in order.
    """
    t_lo, t_hi = between
    i0 = max(0, int(round(t_lo * sr)))
    i1 = min(len(env), int(round(t_hi * sr)))
    if i1 - i0 < int(sr * 2):
        return None

    seg = env[i0:i1].astype(np.float32)

    # (a) quantiles: aggressive → permissive
    idx = _segments_above_quantiles(
        seg, sr, q_list=[0.95, 0.90, 0.85, 0.80], min_len=min_len, max_len=max_len
    )
    if idx is not None:
        return (i0 + idx) / float(sr)

    # (b) stats thresholds: mean + k*std
    idx = _segments_above_stats(
        seg, sr, k_list=[2.0, 1.5, 1.2, 1.0], min_len=min_len, max_len=max_len
    )
    if idx is not None:
        return (i0 + idx) / float(sr)

    # (c) matched filter with ~6 s rectangle
    idx = _matched_filter_rect(seg, sr, dur_s=6.0)
    if idx is not None:
        return (i0 + idx) / float(sr)

    return None


def pick_onsets(
    env: np.ndarray,
    sr: int,
    t0_sec: float,
    expected: int,
    min_len: float,
    max_len: float,
    min_gap: float,
    max_iter: int = 14,
) -> List[Detection]:
    """Iteratively tune quantile threshold to get ~expected segments after t0_sec."""
    start_idx = int(round(t0_sec * sr))
    env2 = env[start_idx:]
    # initial quantile
    q = 0.95
    q_lo, q_hi = 0.80, 0.995
    best: Tuple[int, float, List[Tuple[int, int]]] = (0, q, [])
    for _ in range(max_iter):
        thr = float(np.quantile(env2, q))
        spans = segments_above_threshold(env2, sr, thr, min_len, max_len)
        # enforce min gap between starts
        filtered: List[Tuple[int, int]] = []
        last_start = -1e9
        for s_idx, e_idx in spans:
            t_start = s_idx / sr
            if (t_start - last_start) >= min_gap:
                filtered.append((s_idx, e_idx))
                last_start = t_start
        cnt = len(filtered)
        if not best[2] or abs(cnt - expected) < abs(best[0] - expected):
            best = (cnt, q, filtered)
        if cnt == expected:
            spans = filtered
            break
        # adjust quantile
        if cnt > expected:
            q = min(q_hi, q + 0.02)
        else:
            q = max(q_lo, q - 0.02)
    else:
        spans = best[2]

    # If we still have too many, take the earliest expected; if too few, just return what we found (we’ll report)
    spans = spans[:expected]
    dets: List[Detection] = []
    for s_idx, e_idx in spans:
        onset = (start_idx + s_idx) / sr
        dur = (e_idx - s_idx) / sr
        dets.append(Detection(onset, dur))
    return dets


# ---------- light cross-check ----------


def roi_mean_brightness(
    cap: cv2.VideoCapture, mask: np.ndarray, t_s: float
) -> Optional[float]:
    ok = cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    # average over ROI pixels
    m = cv2.mean(roi, mask=mask)[0]
    return float(m)


def validate_light(
    video: Path,
    meta: dict,
    cfg: dict,
    onsets: List[Detection],
    lead: float,
    trial_dur: float,
    tail: float,
) -> List[dict]:
    """Check that ROI brightness surges in the last 2 s of each tone."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [
            {"light_ok": False, "baseline_mean": None, "flash_max": None}
            for _ in onsets
        ]
    mask = build_roi_mask(meta, cfg)
    results: List[dict] = []
    for det in onsets:
        # Baseline window ~3.0 s after onset (before light): [onset+2.5 .. onset+3.5]
        base_ts = np.linspace(
            det.onset_s + 2.5, det.onset_s + 3.5, num=5, endpoint=True
        )
        base_vals = [roi_mean_brightness(cap, mask, t) for t in base_ts]
        base_vals = [v for v in base_vals if v is not None]
        baseline = float(np.mean(base_vals)) if base_vals else None

        # Flash window ~ last 2 s: [onset+4.1 .. onset+6.0]
        flash_ts = np.linspace(
            det.onset_s + 4.1, det.onset_s + 6.0, num=10, endpoint=True
        )
        flash_vals = [roi_mean_brightness(cap, mask, t) for t in flash_ts]
        flash_vals = [v for v in flash_vals if v is not None]
        flash_max = float(np.max(flash_vals)) if flash_vals else None

        light_ok = None
        if baseline is not None and flash_max is not None:
            # Heuristics: either absolute saturation or clear jump over baseline
            light_ok = (flash_max >= 200.0) or (flash_max >= baseline + 30.0)

        results.append(
            {
                "light_ok": bool(light_ok) if light_ok is not None else False,
                "baseline_mean": baseline,
                "flash_max": flash_max,
            }
        )
    cap.release()
    return results


# ---------- cutting ----------


def cut_ffmpeg(
    src: Path, dst: Path, start_s: float, dur_s: float, fps_out: Optional[float] = None
) -> None:
    ensure_dir(dst.parent)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{dur_s:.3f}",
    ]
    if fps_out is not None and fps_out > 0:
        cmd += ["-r", f"{fps_out:.6f}"]  # enforce CFR at ingest fps
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-y",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


# ---------- main ----------


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.split_audio <config.yaml> <video_path> [--min-start-s 600 --expected 50 --trial-dur 6 --lead 1 --tail 1 --reencode]",
            file=sys.stderr,
        )
        return 2

    cfg_path = Path(sys.argv[1])
    video_path = Path(sys.argv[2])
    # defaults
    min_start_s = 600.0
    expected = 50
    trial_dur = 6.0
    lead = 1.0
    tail = 1.0
    # optional flags
    for i, tok in enumerate(sys.argv):
        if tok == "--min-start-s" and i + 1 < len(sys.argv):
            min_start_s = float(sys.argv[i + 1])
        if tok == "--expected" and i + 1 < len(sys.argv):
            expected = int(sys.argv[i + 1])
        if tok == "--trial-dur" and i + 1 < len(sys.argv):
            trial_dur = float(sys.argv[i + 1])
        if tok == "--lead" and i + 1 < len(sys.argv):
            lead = float(sys.argv[i + 1])
        if tok == "--tail" and i + 1 < len(sys.argv):
            tail = float(sys.argv[i + 1])

    cfg = yaml.safe_load(cfg_path.read_text())
    paths = cfg.get("paths", {})
    out_dir = Path(paths.get("out", "out"))
    logs_dir = Path(paths.get("logs", "logs"))
    events_dir = Path(paths.get("events", "events"))
    ensure_dir(out_dir)
    ensure_dir(logs_dir)
    ensure_dir(events_dir)
    stem = video_path.stem

    ingest = read_ingest(out_dir, stem)
    fps = float(ingest["fps"])
    duration_s = float(
        ingest.get("duration_s") or (ingest["frames"] / fps if fps > 0 else 0.0)
    )

    # 1) Extract audio to WAV (mono, 2000 Hz)
    wav_path = out_dir / "audio" / f"{stem}_mono2000.wav"
    if not wav_path.exists():
        if not ffmpeg_available():
            print(
                "ERROR: ffmpeg not found; required to extract audio.", file=sys.stderr
            )
            return 1
        extract_audio_wav(video_path, wav_path, sr=2000)

    # 2) Read & envelope
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim != 1:
        x = x[:, 0]
    env = bandpass_envelope(x, sr=sr, lo=5.0, hi=80.0, smooth_sec=0.2)

    # 3) Pick onsets (≥600 s, 6 s tone, min gap ~6 s)
    dets = pick_onsets(
        env=env,
        sr=sr,
        t0_sec=min_start_s,
        expected=expected,
        min_len=5.5,
        max_len=7.2,
        min_gap=6.0,
    )
    # If too few, fall back to flash-first and backfill audio-only trials
    use_hybrid = len(dets) < 40
    if use_hybrid:
        flashes = detect_flashes(video_path, ingest, cfg, min_start_s=min_start_s)
        # Convert flashes (paired) to sound onsets by subtracting 4 s
        paired_onsets = [max(min_start_s, f - 4.0) for f in sorted(flashes)]
        # Form 5 blocks of 9 paired trials; infer the 10th (sound_only) via audio envelope in the gap
        blocks = [
            paired_onsets[b * 9 : (b + 1) * 9]
            for b in range(5)
            if len(paired_onsets) >= (b + 1) * 9
        ]
        found_onsets: List[Tuple[float, str, str]] = []
        for b, block in enumerate(blocks):
            # append 9 paired
            for t0 in block:
                found_onsets.append((t0, "paired", "flash->sound-4s"))
            # sound-only search window: between last paired of block and first paired of next block
            if b < len(blocks) - 1:
                lo = block[-1] + 5.5  # can start ~5.5 s after last paired onset
                hi = (
                    blocks[b + 1][0] - 1.0
                )  # must finish before next flash’s light phase
            else:
                lo = block[-1] + 5.5
                hi = duration_s - 2.0  # last block: search until near end-of-video
            so = infer_sound_only_from_audio(env, sr, (lo, hi))
            if so is not None:
                found_onsets.append((so, "sound_only", "audio_gap"))
        # sort and trim to expected 50
        found_onsets.sort(key=lambda x: x[0])
        dets = [Detection(onset_s=fo[0], dur_s=6.0) for fo in found_onsets][:expected]
        labels_by_idx = [fo[1] for fo in found_onsets][: len(dets)]
        methods_by_idx = [fo[2] for fo in found_onsets][: len(dets)]

    else:
        labels_by_idx = [
            "sound_only" if (i % 10 == 0) else "paired" for i in range(1, len(dets) + 1)
        ]
        methods_by_idx = ["audio_plateau"] * len(dets)

    # 4) Build trial table (CSV: onset_s, duration_s, label) + keep methods for QC only
    trials_csv = out_dir / "tables" / f"{stem}_trials.csv"
    ensure_dir(trials_csv.parent)
    rows: List[dict] = []
    methods_for_qc: List[str] = []

    for i, d in enumerate(dets, start=1):
        label = labels_by_idx[i - 1]  # <-- no trailing comma
        method = methods_by_idx[i - 1] if i - 1 < len(methods_by_idx) else "unknown"
        rows.append(
            {
                "onset_s": round(d.onset_s, 3),
                "duration_s": float(trial_dur),
                "label": label,
            }
        )
        methods_for_qc.append(method)

    with trials_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["onset_s", "duration_s", "label"])
        w.writeheader()
        w.writerows(rows)

    # 5) Light cross-check in ROI (final 2 s of tone)
    light_checks = validate_light(
        video=video_path,
        meta=ingest,
        cfg=cfg,
        onsets=dets,
        lead=lead,
        trial_dur=trial_dur,
        tail=tail,
    )
    qc_csv = out_dir / "tables" / f"{stem}_split_qc.csv"
    ensure_dir(qc_csv.parent)
    with qc_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial_index",
                "onset_s",
                "label",
                "method",
                "light_ok",
                "baseline_mean",
                "flash_max",
            ],
        )
        w.writeheader()
        for i, (row, chk) in enumerate(zip(rows, light_checks), start=1):
            w.writerow(
                {
                    "trial_index": i,
                    "onset_s": row["onset_s"],
                    "label": row["label"],
                    "method": methods_for_qc[i - 1]
                    if i - 1 < len(methods_for_qc)
                    else "unknown",
                    **chk,
                }
            )

    # 6) Quick envelope plot (optional; only if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        t = np.arange(len(env)) / float(sr)
        plt.figure(figsize=(10, 3))
        plt.plot(t, env, linewidth=1)
        for r in rows:
            t0 = r["onset_s"]
            plt.axvspan(t0, t0 + trial_dur, alpha=0.15, linewidth=0)
        plt.xlim(
            min_start_s - 5, min(duration_s, min_start_s + 120)
        )  # show first 2 min post-acclimation
        plt.xlabel("Time (s)")
        plt.ylabel("Envelope")
        fig_path = out_dir / "figs" / f"{stem}_audio_envelope.png"
        ensure_dir(fig_path.parent)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    except Exception:
        fig_path = None

    # 7) Cut frame-accurate clips (8 s) with ffmpeg re-encode
    clips_dir = out_dir / "clips" / stem
    ensure_dir(clips_dir)
    for i, r in enumerate(rows, start=1):
        start = max(min_start_s, float(r["onset_s"]) - lead)
        dur = float(lead + trial_dur + tail)
        label = r["label"]
        dst = clips_dir / f"trial_{i:03d}_{label}.mp4"
        cut_ffmpeg(video_path, dst, start, dur, fps_out=fps)

    # 7b) Build events CSV for downstream modules (trial_id, clip_path, cs_frame, us_frame, roi_path)
    # Derive directories and stable CFR fps for frame math
    ensure_dir(events_dir)
    roi_png = export_roi_mask_png(ingest, cfg, out_dir, stem)
    events_csv = events_dir / f"{stem}_trials.csv"

    clip_fps = fps  # we enforced CFR at this fps when cutting

    def _frame_at(seconds_from_clip_start: float) -> int:
        return int(round(seconds_from_clip_start * clip_fps))

    # CS happens 1.0 s into each 8 s clip (lead = 1.0)
    cs_frame_idx = _frame_at(lead)

    with events_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["trial_id", "clip_path", "cs_frame", "us_frame", "roi_path"],
        )
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            trial_id = f"trial_{i:03d}"
            label = r["label"]
            clip_name = f"{trial_id}_{label}.mp4"
            clip_path = str((clips_dir / clip_name).as_posix())
            # US onset is 4.0 s after CS onset for paired trials; blank otherwise
            us_frame_idx = _frame_at(lead + 4.0) if label != "sound_only" else ""
            w.writerow(
                {
                    "trial_id": trial_id,
                    "clip_path": clip_path,
                    "cs_frame": cs_frame_idx,
                    "us_frame": us_frame_idx,
                    "roi_path": str(roi_png.as_posix()),
                }
            )

    # 8) Write split manifest + log
    split_meta = {
        "video": str(video_path),
        "sha256": ingest.get("sha256"),
        "fps": fps,
        "duration_s": duration_s,
        "min_start_s": min_start_s,
        "expected_trials": expected,
        "found_trials": len(rows),
        "trial_duration_s": float(trial_dur),
        "lead_s": float(lead),
        "tail_s": float(tail),
        "roi": cfg.get("detect_track", {}).get("rois", {}),
        "trials_csv": str(trials_csv),
        "qc_csv": str(qc_csv),
        "envelope_png": str(fig_path) if fig_path else None,
        "clips_dir": str(clips_dir),
        "events_csv": str(events_csv),
        "roi_png": str(roi_png),
    }
    write_json(out_dir / "meta" / f"{stem}.split.json", split_meta)
    write_jsonl(
        logs_dir / "split_audio.jsonl",
        [
            {
                "stage": "split_audio",
                "video": str(video_path),
                "level": "INFO",
                "msg": "ok",
                "found_trials": len(rows),
                "expected": expected,
            }
        ],
    )

    # Sanity print
    print(f"[split_audio] Found {len(rows)} trials; wrote clips to {clips_dir}")
    print(f"[split_audio] Trials table → {trials_csv}")
    print(f"[split_audio] QC table     → {qc_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
