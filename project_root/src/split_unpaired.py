#!/usr/bin/env python3
"""
Unpaired-aware trial splitter for WormWatcher.

Given a schedule (CSV or grid), for each scheduled onset:
 - Detects audio (~6 s tone) from 5–80 Hz band-pass + Hilbert envelope.
 - Detects light as a luminance surge in the ROI.
 - Classifies the trial as one of:
     paired        : audio & light aligned (light ≈ audio_onset + expected_light_offset ± pair_tol)
     both_unpaired : audio & light present but not aligned
     sound_only    : only audio
     light_only    : only light
     none          : neither detected
 - Cuts frame-accurate clips (1 s lead + 6 s dur + 1 s tail).

Outputs:
  out/tables/<stem>_trials.csv
  out/tables/<stem>_split_qc.csv
  out/meta/<stem>.split.json
  out/clips/<stem>/trial_###_<label>.mp4
  logs/split_audio.jsonl

Usage examples:
  # CSV schedule (column: onset_s)
  python -m src.split_unpaired config.yaml videos/D5W1P28Jun25.m4v \
    --schedule-csv events/sched.csv --min-start-s 600 --trial-dur 6 --lead 1 --tail 1 \
    --pre-win 2 --post-win 8 --pair-tol 0.6 --expected-light-offset 4.0

  # Grid schedule (t0 + k*period, k=0..count-1)
  python -m src.split_unpaired config.yaml videos/D5W1P28Jun25.m4v \
    --grid-t0 611 --grid-period 12 --grid-count 50 \
    --min-start-s 600 --trial-dur 6 --lead 1 --tail 1 --pre-win 2 --post-win 8 --pair-tol 0.6
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import csv
import json
import shutil
import subprocess
import sys

import cv2
import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, filtfilt, hilbert


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
    """Extract mono WAV at sr Hz via ffmpeg."""
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
        if min_len <= length_s <= max_len:
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
    z = (seg - seg.mean()) / (seg.std() or 1.0)
    k = np.ones(n, dtype=np.float32) / float(n)
    corr = np.convolve(z, k, mode="valid")
    peak = int(np.argmax(corr))
    return peak


def find_audio_onset_in_window(
    env: np.ndarray,
    sr: int,
    window: Tuple[float, float],
    min_len: float = 5.3,
    max_len: float = 7.5,
) -> Optional[float]:
    """Search env between (t_lo, t_hi) for one ~6 s plateau; return onset or None."""
    t_lo, t_hi = window
    i0 = max(0, int(round(t_lo * sr)))
    i1 = min(len(env), int(round(t_hi * sr)))
    if i1 - i0 < int(sr * 2):
        return None
    seg = env[i0:i1].astype(np.float32)

    idx = _segments_above_quantiles(
        seg, sr, q_list=[0.95, 0.90, 0.85, 0.80], min_len=min_len, max_len=max_len
    )
    if idx is not None:
        return (i0 + idx) / float(sr)

    idx = _segments_above_stats(
        seg, sr, k_list=[2.0, 1.5, 1.2, 1.0], min_len=min_len, max_len=max_len
    )
    if idx is not None:
        return (i0 + idx) / float(sr)

    idx = _matched_filter_rect(seg, sr, dur_s=6.0)
    if idx is not None:
        return (i0 + idx) / float(sr)

    return None


# ---------- light detection ----------


def find_light_onset_in_window(
    video: Path,
    meta: dict,
    cfg: dict,
    window: Tuple[float, float],
    sample_hz: float = 20.0,
    smooth_win: int = 5,
    q: float = 0.95,
) -> Optional[float]:
    """Scan [t_lo, t_hi] for a luminance surge in the ROI; return onset time or None."""
    t_lo, t_hi = window
    if t_hi <= t_lo:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    mask = build_roi_mask(meta, cfg)
    float(meta["fps"])
    step = 1.0 / max(1.0, sample_hz)
    ts = np.arange(t_lo, t_hi, step, dtype=np.float32)
    vals: List[float] = []
    for t in ts:
        _ = cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            vals.append(np.nan)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(cv2.mean(gray, mask=mask)[0])
    cap.release()
    v = np.array([x for x in vals if not np.isnan(x)], dtype=np.float32)
    if v.size < 3:
        return None
    if smooth_win > 1 and (smooth_win % 2 == 1):
        pad = smooth_win // 2
        v = np.convolve(
            np.pad(v, (pad, pad), mode="edge"),
            np.ones(smooth_win, np.float32) / smooth_win,
            mode="valid",
        )
    thr = float(np.quantile(v, q))
    above = (v >= thr).astype(np.uint8)
    rise = np.where((above[1:] == 1) & (above[:-1] == 0))[0] + 1
    if rise.size == 0:
        return None
    idx = int(rise[0])
    return float(ts[min(idx, len(ts) - 1)])


# ---------- classification ----------


def classify_trial_window(
    env: np.ndarray,
    sr: int,
    video: Path,
    meta: dict,
    cfg: dict,
    t_sched: float,
    pre_win: float = 2.0,
    post_win: float = 8.0,
    expected_light_offset: float = 4.0,
    pair_tol_s: float = 0.6,
) -> Tuple[float, str, str, Optional[float], Optional[float]]:
    """
    Inspect a window around scheduled onset and classify:
      paired          : audio & light and |(tL - tA) - expected_light_offset| <= pair_tol_s
      both_unpaired   : audio & light but not aligned
      sound_only      : audio only
      light_only      : light only
      none            : neither detected
    Returns: (onset_used, label, method, t_audio, t_light)
      onset_used anchors clipping (prefer audio onset; else light onset; else t_sched).
    """
    t_lo = max(0.0, t_sched - pre_win)
    t_hi = t_sched + post_win

    t_audio = find_audio_onset_in_window(
        env, sr, (t_lo, t_hi), min_len=5.3, max_len=7.5
    )
    t_light = find_light_onset_in_window(video, meta, cfg, (t_lo, t_hi))

    if (t_audio is not None) and (t_light is not None):
        offset = t_light - t_audio
        if abs(offset - expected_light_offset) <= pair_tol_s:
            return (t_audio, "paired", "schedule+audio+light", t_audio, t_light)
        return (t_audio, "both_unpaired", "schedule+audio+light", t_audio, t_light)

    if t_audio is not None:
        return (t_audio, "sound_only", "schedule+audio", t_audio, t_light)

    if t_light is not None:
        return (t_light, "light_only", "schedule+light", t_audio, t_light)

    return (t_sched, "none", "schedule_only", t_audio, t_light)


# ---------- cutting ----------


def cut_ffmpeg(src: Path, dst: Path, start_s: float, dur_s: float) -> None:
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
            "Usage: python -m src.split_unpaired <config.yaml> <video_path> "
            "[--min-start-s 600 --trial-dur 6 --lead 1 --tail 1 "
            "--schedule-csv events/sched.csv | --grid-t0 T0 --grid-period P --grid-count N "
            "--pre-win 2 --post-win 8 --pair-tol 0.6 --expected-light-offset 4.0]",
            file=sys.stderr,
        )
        return 2

    cfg_path = Path(sys.argv[1])
    video_path = Path(sys.argv[2])

    # defaults
    min_start_s = 600.0
    trial_dur = 6.0
    lead = 1.0
    tail = 1.0
    pre_win = 2.0
    post_win = 8.0
    pair_tol_s = 0.6
    expected_light_offset = 4.0
    schedule_csv: Optional[str] = None
    grid_t0: Optional[float] = None
    grid_period: Optional[float] = None
    grid_count: Optional[int] = None

    # parse flags (simple hand-rolled)
    for i, tok in enumerate(sys.argv):
        if tok == "--min-start-s" and i + 1 < len(sys.argv):
            min_start_s = float(sys.argv[i + 1])
        if tok == "--trial-dur" and i + 1 < len(sys.argv):
            trial_dur = float(sys.argv[i + 1])
        if tok == "--lead" and i + 1 < len(sys.argv):
            lead = float(sys.argv[i + 1])
        if tok == "--tail" and i + 1 < len(sys.argv):
            tail = float(sys.argv[i + 1])

        if tok == "--schedule-csv" and i + 1 < len(sys.argv):
            schedule_csv = sys.argv[i + 1]
        if tok == "--grid-t0" and i + 1 < len(sys.argv):
            grid_t0 = float(sys.argv[i + 1])
        if tok == "--grid-period" and i + 1 < len(sys.argv):
            grid_period = float(sys.argv[i + 1])
        if tok == "--grid-count" and i + 1 < len(sys.argv):
            grid_count = int(sys.argv[i + 1])

        if tok == "--pre-win" and i + 1 < len(sys.argv):
            pre_win = float(sys.argv[i + 1])
        if tok == "--post-win" and i + 1 < len(sys.argv):
            post_win = float(sys.argv[i + 1])
        if tok == "--pair-tol" and i + 1 < len(sys.argv):
            pair_tol_s = float(sys.argv[i + 1])
        if tok == "--expected-light-offset" and i + 1 < len(sys.argv):
            expected_light_offset = float(sys.argv[i + 1])

    cfg = yaml.safe_load(cfg_path.read_text())
    paths = cfg["paths"]
    out_dir = Path(paths["out"])
    Path(paths["logs"])
    stem = video_path.stem

    ingest = read_ingest(out_dir, stem)
    fps = float(ingest["fps"])
    duration_s = float(
        ingest.get("duration_s") or (ingest["frames"] / fps if fps > 0 else 0.0)
    )

    # Ensure audio WAV exists (mono 2 kHz)
    wav_path = out_dir / "audio" / f"{stem}_mono2000.wav"
    if not wav_path.exists():
        if not ffmpeg_available():
            print(
                "ERROR: ffmpeg not found; required to extract audio.", file=sys.stderr
            )
            return 1
        extract_audio_wav(video_path, wav_path, sr=2000)

    # Read audio and compute envelope
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim != 1:
        x = x[:, 0]
    env = bandpass_envelope(x, sr=sr, lo=5.0, hi=80.0, smooth_sec=0.2)

    # Build schedule list
    if schedule_csv:
        import pandas as _pd

        sched_df = _pd.read_csv(schedule_csv)
        sched_onsets = [
            float(t) for t in sched_df["onset_s"].tolist() if float(t) >= min_start_s
        ]
    elif (
        (grid_t0 is not None) and (grid_period is not None) and (grid_count is not None)
    ):
        sched_onsets = [grid_t0 + i * grid_period for i in range(grid_count)]
        sched_onsets = [t for t in sched_onsets if t >= min_start_s]
    else:
        print(
            "ERROR: provide --schedule-csv or --grid-t0/--grid-period/--grid-count.",
            file=sys.stderr,
        )
        return 2

    # Classify each scheduled window
    trials_rows: List[dict] = []
    methods_for_qc: List[str] = []
    t_audio_list: List[Optional[float]] = []
    t_light_list: List[Optional[float]] = []
    used_onsets: List[Detection] = []

    for idx, t_sched in enumerate(sched_onsets, start=1):
        t_used, label, method, t_audio, t_light = classify_trial_window(
            env=env,
            sr=sr,
            video=video_path,
            meta=ingest,
            cfg=cfg,
            t_sched=t_sched,
            pre_win=pre_win,
            post_win=post_win,
            expected_light_offset=expected_light_offset,
            pair_tol_s=pair_tol_s,
        )
        onset_used = float(t_used)
        trials_rows.append(
            {
                "onset_s": round(onset_used, 3),
                "duration_s": float(trial_dur),
                "label": label,
            }
        )
        methods_for_qc.append(method)
        t_audio_list.append(t_audio)
        t_light_list.append(t_light)
        used_onsets.append(Detection(onset_used, trial_dur))

    # Write trials CSV
    trials_csv = out_dir / "tables" / f"{stem}_trials.csv"
    ensure_dir(trials_csv.parent)
    with trials_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["onset_s", "duration_s", "label"])
        w.writeheader()
        w.writerows(trials_rows)

    # Light QC (reuse validator for consistency)
    light_checks = validate_light(
        video=video_path,
        meta=ingest,
        cfg=cfg,
        onsets=used_onsets,
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
                "t_sched",
                "onset_s",
                "label",
                "method",
                "t_audio",
                "t_light",
                "light_ok",
                "baseline_mean",
                "flash_max",
            ],
        )
        w.writeheader()
        for i, (sched_t, row, chk, ta, tl, meth) in enumerate(
            zip(
                sched_onsets,
                trials_rows,
                light_checks,
                t_audio_list,
                t_light_list,
                methods_for_qc,
            ),
            start=1,
        ):
            w.writerow(
                {
                    "trial_index": i,
                    "t_sched": round(float(sched_t), 3),
                    "onset_s": row["onset_s"],
                    "label": row["label"],
                    "method": meth,
                    "t_audio": None if ta is None else round(float(ta), 3),
                    "t_light": None if tl is None else round(float(tl), 3),
                    **chk,
                }
            )

    # Cut frame-accurate 8 s clips (re-encode)
    clips_dir = out_dir / "clips" / stem
    ensure_dir(clips_dir)
    for i, r in enumerate(trials_rows, start=1):
        start = max(min_start_s, float(r["onset_s"]) - lead)
        dur = float(lead + trial_dur + tail)
        label = r["label"]
        dst = clips_dir / f"trial_{i:03d}_{label}.mp4"
        cut_ffmpeg(video_path, dst, start, dur)

    # Manifest + simple log
    split_meta = {
        "video": str(video_path),
        "sha256": ingest.get("sha256"),
        "fps": fps,
        "duration_s": duration_s,
        "min_start_s": min_start_s,
        "scheduled_trials": len(sched_onsets),
        "found_trials": len(trials_rows),
        "trial_duration_s": float(trial_dur),
        "lead_s": float(lead),
        "tail_s": float(tail),
        "roi": cfg.get("detect_track", {}).get("rois", {}),
        "trials_csv": str(trials_csv),
        "qc_csv": str(qc_csv),
        "clips_dir": str(clips_dir),
        "mode": "schedule",
        "pair_tol_s": pair_tol_s,
        "expected_light_offset": expected_light_offset,
        "pre_win": pre_win,
        "post_win": post_win,
    }
    write_json(out_dir / "meta" / f"{stem}.split.json", split_meta)
    write_jsonl(
        Path(paths["logs"]) / "split_audio.jsonl",
        [
            {
                "stage": "split_unpaired",
                "video": str(video_path),
                "level": "INFO",
                "msg": "ok",
                "scheduled": len(sched_onsets),
                "found": len(trials_rows),
            }
        ],
    )

    print(f"[split_unpaired] Wrote {len(trials_rows)} trials; clips → {clips_dir}")
    print(f"[split_unpaired] Trials table → {trials_csv}")
    print(f"[split_unpaired] QC table     → {qc_csv}")
    return 0


# ---------- light QC reused from audio splitter ----------


def roi_mean_brightness(
    cap: cv2.VideoCapture, mask: np.ndarray, t_s: float
) -> Optional[float]:
    _ = cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
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
        base_ts = np.linspace(
            det.onset_s + 2.5, det.onset_s + 3.5, num=5, endpoint=True
        )
        base_vals = [roi_mean_brightness(cap, mask, t) for t in base_ts]
        base_vals = [v for v in base_vals if v is not None]
        baseline = float(np.mean(base_vals)) if base_vals else None

        flash_ts = np.linspace(
            det.onset_s + 4.1, det.onset_s + 6.0, num=10, endpoint=True
        )
        flash_vals = [roi_mean_brightness(cap, mask, t) for t in flash_ts]
        flash_vals = [v for v in flash_vals if v is not None]
        flash_max = float(np.max(flash_vals)) if flash_vals else None

        light_ok = None
        if baseline is not None and flash_max is not None:
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


if __name__ == "__main__":
    raise SystemExit(main())
