"""
Tierpsy-style detect/track with multi-worker support, robust overlays, and
skeleton + clitellum + head/tail orientation.

Deps: opencv-python, numpy, pandas, pyarrow, scikit-image, scipy, tqdm, pyyaml
Run (from project_root):
    python -m src.detect_track config.tierpsy.yaml --list-trials
    python -m src.detect_track config.tierpsy.yaml --only T001
    python -m src.detect_track config.tierpsy.yaml --workers 3
"""

# --- imports ---
import os
import sys
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- dual-run import bootstrap (supports "python -m" and Spyder %runfile) ---
if __package__ in (None, ""):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(this_dir)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from src.segmentation import segment_frame
    from src.skeletonize import make_skeleton
    from src.clitellum import distance_profile, find_clitellum_index
    from src.orientation import OrientationFilter
else:
    from .segmentation import segment_frame
    from .skeletonize import make_skeleton
    from .clitellum import distance_profile, find_clitellum_index
    from .orientation import OrientationFilter
# ---------------------------------------------------------------------------


# ---------------- helpers ----------------
def load_yaml(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_rect_mask_from_frac(shape_hw, rect_frac):
    """rect_frac = [x0, y0, w, h] in fractions of (W,H)"""
    H, W = shape_hw
    x0 = int(max(0, min(W - 1, round(rect_frac[0] * W))))
    y0 = int(max(0, min(H - 1, round(rect_frac[1] * H))))
    w = int(max(1, round(rect_frac[2] * W)))
    h = int(max(1, round(rect_frac[3] * H)))
    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m, (x0, y0, x1, y1)


def ensure_dirs(cfg):
    out = Path(cfg["paths"]["out"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "trials").mkdir(parents=True, exist_ok=True)
    (out / "qc" / "overlay").mkdir(parents=True, exist_ok=True)
    (out / "qc" / "snapshots").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def _resolve_path(p: str, root: str | None) -> str:
    if not p:
        return p
    P = Path(p)
    if P.is_absolute():
        return str(P)
    if root:
        return str(Path(root) / P)
    return str(P)


def _get_float(row, key, default=None):
    if key is None:
        return default
    try:
        v = row.get(key, default)
        if v is None:
            return default
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return default
        return float(v)
    except Exception:
        return default


def _sanitize_fps(raw_fps):
    try:
        fps = float(raw_fps)
    except Exception:
        return 30.0
    if not math.isfinite(fps) or fps < 1 or fps > 240:
        return 30.0
    return fps


def _make_writer(cfg, base_path, fps, size_hw):
    """Create a VideoWriter with fallbacks; returns (writer, out_path)."""
    h, w = size_hw
    qc = cfg.get("qc", {})
    ext = str(qc.get("overlay_ext", "avi")).lower()
    codec = str(qc.get("overlay_codec", "mjpg")).lower()
    codec_map = {
        "mjpg": cv2.VideoWriter_fourcc(*"MJPG"),
        "mp4v": cv2.VideoWriter_fourcc(*"mp4v"),
        "avc1": cv2.VideoWriter_fourcc(*"avc1"),
        "xvid": cv2.VideoWriter_fourcc(*"XVID"),
    }
    fourcc = codec_map.get(codec, cv2.VideoWriter_fourcc(*"MJPG"))
    out_path = f"{base_path}.{ext}"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        # fallback to MJPG/AVI
        out_path = f"{base_path}.avi"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    return writer, out_path


def _load_roi_mask_for_row(row, cfg):
    """Load ROI from per-row column or default; apply regardless of 'roi.apply' if a per-row ROI exists."""
    roi_col = cfg.get("roi_col")
    roi_path = None
    if roi_col and roi_col in row and isinstance(row[roi_col], str) and row[roi_col]:
        roi_path = row[roi_col]
    elif cfg.get("roi", {}).get("apply", False) and cfg.get("roi_default"):
        roi_path = cfg["roi_default"]
    if not roi_path:
        return None
    roi_path = _resolve_path(roi_path, cfg.get("rois_root"))
    if Path(roi_path).exists():
        m = cv2.imread(roi_path, 0)
        return (m > 0).astype(np.uint8)
    return None


# ----------------------------------------


# ----------- CSV loader (clips or windows mode) -----------
def read_trials(cfg):
    df = pd.read_csv(cfg["events_csv"])

    original_cols = list(df.columns)
    norm_map = {c.lower().strip(): c for c in original_cols}

    def pick(candidates, required=True, label=""):
        for cand in candidates:
            if not cand:
                continue
            key = str(cand).lower().strip()
            if key in norm_map:
                return norm_map[key]
        if required:
            raise ValueError(
                f"Missing required column for {label}. "
                f"Tried {candidates}. CSV has: {original_cols}"
            )
        return None

    # Try "clips mode" (has a clip/path-like column).
    clip_candidates = [
        cfg.get("clips_col"),
        "clip_path",
        "path",
        "video",
        "video_path",
        "filename",
        "file",
        "clip",
        "movie",
        "movie_path",
        "dst_path",
        "out_path",
        "qc_clip",
        "clipfile",
        "relpath",
    ]
    clip_col = pick(clip_candidates, required=False, label="clip_path")

    # trial id / label
    trial_id_candidates = [
        cfg.get("trial_id_col"),
        "trial_id",
        "trial",
        "trialid",
        "trialnum",
        "trial_no",
        "trialname",
        "trial_name",
        "id",
        "name",
        "label",
        "clip_id",
        "clip_name",
    ]
    trial_col = pick(trial_id_candidates, required=False, label="trial_id")

    # optional
    cs_candidates = [
        cfg.get("cs_frame_col"),
        "cs_frame",
        "cs",
        "cs_on",
        "cs_index",
        "cs_start",
        "csframe",
        "cs_start_frame",
        "csframe_idx",
    ]
    us_candidates = [
        cfg.get("us_frame_col"),
        "us_frame",
        "us",
        "us_on",
        "us_index",
        "us_start",
        "usframe",
        "us_start_frame",
        "usframe_idx",
    ]
    roi_candidates = [
        cfg.get("roi_col"),
        "roi_path",
        "roi",
        "mask",
        "roi_file",
        "mask_path",
    ]

    cs_col = pick(cs_candidates, required=False)
    us_col = pick(us_candidates, required=False)
    roi_col = pick(roi_candidates, required=False)

    # CLIPS MODE
    if clip_col is not None:
        if trial_col is None:
            df["trial_id"] = [f"T{ix:03d}" for ix in range(1, len(df) + 1)]
            trial_col = "trial_id"
        rename = {trial_col: "trial_id", clip_col: "clip_path"}
        if cs_col:
            rename[cs_col] = "cs_frame"
        if us_col:
            rename[us_col] = "us_frame"
        if roi_col:
            rename[roi_col] = "roi_path"
        df = df.rename(columns=rename)
        df["trial_id"] = df["trial_id"].astype(str)
        cfg["trial_id_col"] = "trial_id"
        cfg["clips_col"] = "clip_path"
        if "cs_frame" in df.columns:
            cfg["cs_frame_col"] = "cs_frame"
        if "us_frame" in df.columns:
            cfg["us_frame_col"] = "us_frame"
        if "roi_path" in df.columns:
            cfg["roi_col"] = "roi_path"
        cfg["mode"] = "clips"
        print(f"[read_trials] mode=clips mapped -> {list(df.columns)} (rows={len(df)})")
        return df

    # WINDOWS MODE (events with onset/duration or start/end)
    onset_candidates = [
        "onset_s",
        "start_s",
        "start_sec",
        "t0",
        "start_time_s",
        "cs_onset_s",
    ]
    dur_candidates = ["duration_s", "dur_s", "duration", "length_s"]
    end_candidates = ["end_s", "stop_s", "end_sec", "t1", "end_time_s"]

    onset_col = pick(onset_candidates, required=False, label="onset_s")
    dur_col = pick(dur_candidates, required=False, label="duration_s")
    end_col = pick(end_candidates, required=False, label="end_s")

    if onset_col is None or (dur_col is None and end_col is None):
        raise ValueError(
            "Could not detect clips nor window columns. "
            f"Tried clip candidates: {clip_candidates}. "
            f"Tried onset/duration candidates: onset={onset_candidates}, dur={dur_candidates}, end={end_candidates}. "
            f"CSV has: {original_cols}"
        )

    # --- Build canonical columns for windows mode ---
    df = df.copy()

    # Always synthesize deterministic trial IDs so --only is predictable
    id_cfg = cfg.get("id", {}) if isinstance(cfg.get("id"), dict) else {}
    prefix = str(id_cfg.get("prefix", "T"))
    digits = int(id_cfg.get("digits", 3))
    start_ix = int(id_cfg.get("start", 1))

    df["trial_id"] = [
        f"{prefix}{i:0{digits}d}" for i in range(start_ix, start_ix + len(df))
    ]

    # If a nice label column exists, keep it (optional, for reference)
    label_col = "label" if "label" in df.columns else None
    if label_col:
        df.rename(columns={label_col: "label"}, inplace=True)  # ensure exact spelling

    # store optional columns; actual frame math happens in process_trial
    if cs_col:
        df = df.rename(columns={cs_col: "cs_frame"})
    if us_col:
        df = df.rename(columns={us_col: "us_frame"})
    if roi_col:
        df = df.rename(columns={roi_col: "roi_path"})

    # update cfg with canonical names + mode
    cfg["trial_id_col"] = "trial_id"
    cfg["mode"] = "windows"
    cfg["_onset_col"] = onset_col
    cfg["_dur_col"] = dur_col
    cfg["_end_col"] = end_col
    if "roi_path" in df.columns:
        cfg["roi_col"] = "roi_path"
    if "cs_frame" in df.columns:
        cfg["cs_frame_col"] = "cs_frame"
    if "us_frame" in df.columns:
        cfg["us_frame_col"] = "us_frame"

    if not cfg.get("source_video"):
        raise ValueError(
            "Windows mode requires `source_video` in config (path to the base video)."
        )

    print(
        f"[read_trials] mode=windows using source_video={cfg['source_video']} "
        f"-> cols {list(df.columns)} (rows={len(df)}) "
        f"| ids {df['trial_id'].iloc[0]}..{df['trial_id'].iloc[-1]}"
    )
    return df


# ----------------------------------------------------------


def process_trial(row, cfg):
    trial_id = str(row[cfg["trial_id_col"]])
    mode = cfg.get("mode", "clips")

    # --- open capture & window bounds ---
    if mode == "clips":
        clip_path = _resolve_path(row[cfg["clips_col"]], cfg.get("clips_root"))
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return (
                trial_id,
                {"status": "error", "msg": f"cannot_open:{clip_path}"},
                None,
            )
        raw_fps = cfg.get("fps_override") or cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = _sanitize_fps(raw_fps)
        start_frame = 0
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_for_log = clip_path
    else:
        base = _resolve_path(cfg["source_video"], None)
        cap = cv2.VideoCapture(base)
        if not cap.isOpened():
            return (
                trial_id,
                {"status": "error", "msg": f"cannot_open_source:{base}"},
                None,
            )
        raw_fps = cfg.get("fps_override") or cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = _sanitize_fps(raw_fps)

        onset_col = cfg.get("_onset_col")
        dur_col = cfg.get("_dur_col")
        end_col = cfg.get("_end_col")

        onset_s = _get_float(row, onset_col, 0.0)
        if dur_col:
            duration_s = _get_float(row, dur_col, None)
            if duration_s is None and end_col:
                end_s = _get_float(row, end_col, None)
                duration_s = (
                    None if end_s is None else max(0.0, end_s - (onset_s or 0.0))
                )
        else:
            end_s = _get_float(row, end_col, 0.0)
            duration_s = max(0.0, end_s - (onset_s or 0.0))

        start_frame = max(0, int(round((onset_s or 0.0) * fps)))
        frames_to_read = int(round((duration_s or 0.0) * fps))
        end_frame = (
            start_frame + frames_to_read
            if frames_to_read > 0
            else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        source_for_log = f"{base}[{start_frame}:{end_frame}]"

    out_dir = Path(cfg["paths"]["out"])

    # --- background model from the start of this window ---
    bg = None
    seg_cfg = cfg.get("seg", {})
    bg_n = int(seg_cfg.get("bg_frames", 0))
    if bg_n > 0:
        frames = []
        # read up to bg_n frames (but don't overrun window)
        for _ in range(min(bg_n, max(1, end_frame - start_frame))):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if frames:
            bg = np.median(np.stack(frames, 0), 0).astype(np.uint8)
        # restore to window start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    roi_mask = _load_roi_mask_for_row(row, cfg)
    ofilt = OrientationFilter(
        window_frames=cfg["orientation"]["window_frames"],
        flip_hysteresis=cfg["orientation"]["flip_hysteresis"],
        tip_curv_win=cfg["orientation"]["tip_curv_win"],
        clitellum_bias=cfg["orientation"]["clitellum_bias"],
    )

    tracks = []
    prev_path = None
    frame_idx = 0

    qc = cfg.get("qc", {})
    snapshots_on = bool(qc.get("snapshots", True))
    max_snaps = int(qc.get("snapshot_n", 5))
    writer = None
    written = 0

    # --- read/process loop ---
    bbox_roi = None
    while True:
        cur_abs = start_frame + frame_idx
        if cur_abs >= end_frame:
            break
        ok, fr = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if roi_mask is None:
            roi_cfg = cfg.get("roi", {})
            rect_frac = roi_cfg.get("rect_frac")
            if rect_frac:
                roi_mask, bbox_roi = _make_rect_mask_from_frac(gray.shape, rect_frac)
        mask = segment_frame(gray, bg, cfg, roi_mask)
        skel, path = make_skeleton(
            mask,
            cfg["skeleton"]["method"],
            int(cfg["skeleton"]["spur_px"]),
            int(cfg["skeleton"]["bridge_gaps_px"]),
        )

        # clitellum
        clit_idx = None
        if len(path) >= 5:
            radii = distance_profile(mask, path)
            clit_idx = find_clitellum_index(
                radii,
                tuple(cfg["clitellum"]["search_range_pct"]),
                int(cfg["clitellum"]["smooth_win"]),
                float(cfg["clitellum"]["min_prominence_px"]),
            )

        # head/tail orientation
        head_pos = None
        tail_pos = None
        if len(path) >= 2:
            head_end = ofilt.choose_head(path, prev_path, clit_idx)
            if head_end == -1:
                path = list(reversed(path))
            head_pos = path[0]
            tail_pos = path[-1]
        prev_path = path

        # track row
        if head_pos and tail_pos:
            tracks.append(
                {
                    "trial_id": trial_id,
                    "frame": frame_idx,
                    "x_head": head_pos[1],
                    "y_head": head_pos[0],
                    "x_tail": tail_pos[1],
                    "y_tail": tail_pos[0],
                    "clit_idx": -1 if clit_idx is None else clit_idx,
                    "body_len": len(path),
                }
            )

        # overlay setup
        if writer is None and qc.get("overlay", True):
            h, w = fr.shape[:2]
            base = out_dir / "qc" / "overlay" / f"{trial_id}_overlay"
            writer, overlay_path = _make_writer(
                cfg, str(base), _sanitize_fps(cfg.get("fps_override") or fps), (h, w)
            )

        # draw overlay (and snapshots)
        if writer is not None and writer.isOpened():
            ov = fr.copy()
            if mask is not None and mask.any():
                ov[mask > 0] = (
                    0.6 * ov[mask > 0] + 0.4 * np.array([0, 255, 0])
                ).astype(np.uint8)
            if len(path) > 0:
                step = max(1, len(path) // 150)
                for y, x in path[::step]:
                    cv2.circle(ov, (x, y), 1, (0, 0, 255), -1)
            if bbox_roi and cfg.get("qc", {}).get("draw_roi", True):
                x0, y0, x1, y1 = bbox_roi
                cv2.rectangle(
                    ov, (x0, y0), (x1, y1), (255, 255, 0), 1
                )  # yellow ROI box
            if clit_idx is not None and len(path) > clit_idx:
                cy, cx = path[clit_idx]
                cv2.circle(ov, (cx, cy), 3, (255, 0, 0), -1)  # blue: clitellum
            if head_pos:
                cv2.circle(
                    ov, (head_pos[1], head_pos[0]), 4, (0, 0, 0), -1
                )  # black head
            if tail_pos:
                cv2.circle(
                    ov, (tail_pos[1], tail_pos[0]), 3, (255, 255, 255), -1
                )  # white tail
            writer.write(ov)
            written += 1
            if snapshots_on and frame_idx < max_snaps:
                sp = out_dir / "qc" / "snapshots" / f"{trial_id}_f{frame_idx:04d}.png"
                cv2.imwrite(str(sp), ov)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # save per-trial parquet
    tdf = pd.DataFrame(tracks)
    tpath = out_dir / "tables" / "trials" / f"{trial_id}.parquet"
    if len(tdf):
        tdf.to_parquet(tpath, index=False)

    meta = {
        "status": "ok",
        "frames": frame_idx,
        "written": written,
        "fps": float(fps),
        "source": source_for_log,
    }
    if writer is None:
        meta["warn"] = "overlay_disabled_or_writer_not_opened"
    elif written == 0:
        meta["error"] = "no_frames_written_check_codec_fps"
    return trial_id, meta, str(tpath if len(tdf) else "")


def merge_tables(cfg):
    trials_dir = Path(cfg["paths"]["out"]) / "tables" / "trials"
    files = sorted(trials_dir.glob("*.parquet"))
    if not files:
        return
    dfs = [pd.read_parquet(f) for f in files if f.stat().st_size > 0]
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        (Path(cfg["paths"]["out"]) / "tables" / "tracks.parquet").write_bytes(
            b""
        )  # touch
        all_df.to_parquet(
            Path(cfg["paths"]["out"]) / "tables" / "tracks.parquet", index=False
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    ap.add_argument("--only", nargs="*", help="trial ids to process")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--list-trials", action="store_true", help="List recognized trial IDs and exit"
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_dirs(cfg)
    df = read_trials(cfg)

    if args.only:
        df = df[df[cfg["trial_id_col"]].astype(str).isin(set(args.only))]
    if cfg["trial_id_col"] not in df.columns:
        raise ValueError(
            f"trial_id column '{cfg['trial_id_col']}' not in CSV. Columns: {list(df.columns)}"
        )

    if args.list_trials:
        ids = df[cfg["trial_id_col"]].astype(str).tolist()
        print("Recognized trial IDs (first 100):", ids[:100])
        return

    trials = df.to_dict("records")
    workers = args.workers or int(cfg.get("workers", 1))

    results = []
    if workers <= 1:
        for r in tqdm(trials, desc="trials"):
            results.append(process_trial(r, cfg))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_trial, r, cfg): r for r in trials}
            for fut in tqdm(
                as_completed(futs), total=len(futs), desc=f"{workers} workers"
            ):
                results.append(fut.result())

    with open("logs/detect_track.jsonl", "a") as f:
        for tid, meta, _ in results:
            f.write(json.dumps({"trial_id": tid, **meta}) + "\n")

    merge_tables(cfg)
    print("done.")


if __name__ == "__main__":
    main()
