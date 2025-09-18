#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine per-trial clips, inserting a 45-frame (configurable) white "flash" card
that displays the trial number (and optional label) before each clip.

Changelog:
  v0.1.1
   - FIX: concat list now uses ABSOLUTE paths wrapped in single quotes,
     avoiding duplicated prefixes like ".../__tmp__/.../__tmp__/...".
   - Adds --debug to print temp paths and the concat list.

Usage (example):
  python combine_clips_with_flash.py \
      --clips-dir out/clips/D5W1P28Jun25 \
      --output out/qc/D5W1P28Jun25_with_titles.mp4 \
      --frames 45 \
      --pattern "trial_*.mp4" \
      --font-file "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

Requirements:
  - Python 3.10+
  - ffmpeg (with drawtext filter) available on PATH

Notes:
  - FPS, width, and height are inferred from the FIRST input clip and used as the
    standard for preroll generation and final encode to avoid mismatches.
  - Filenames are expected like: trial_001_CS.mp4 or trial_001.mp4
    (trial number required; label optional).

Outputs:
  - The concatenated mp4 written to --output
  - A run log at logs/concat.jsonl (one line JSON with provenance)
  - A temp directory with preroll segments (deleted on --clean)

Author: WormWatcher (HWVA)
License: MIT
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

TRIAL_RE = re.compile(r"trial_(\d+)(?:_([^.]+))?\.mp4$", re.IGNORECASE)


def run(cmd: List[str]) -> None:
    """Run a subprocess, raising on nonzero exit."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(shlex.quote(c) for c in cmd)}")


def ffprobe_json(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout, file=sys.stderr)
        raise RuntimeError("ffprobe failed")
    return json.loads(p.stdout)


def parse_fps(rate: str) -> float:
    # rate like "30000/1001" or "30/1" or "30"
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den)
    return float(rate)


def get_video_params(path: Path) -> Tuple[int, int, float]:
    meta = ffprobe_json(path)
    vstreams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError(f"No video stream found in {path}")
    vs = vstreams[0]
    width = int(vs["width"])
    height = int(vs["height"])
    fps = parse_fps(vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "30")
    return width, height, fps


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_clips(clips_dir: Path, pattern: str) -> List[Path]:
    files = sorted(clips_dir.glob(pattern), key=lambda p: natural_key(p.name))
    return [p for p in files if p.is_file()]


def parse_trial_and_label(name: str) -> Tuple[int, Optional[str]]:
    m = TRIAL_RE.search(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {name}")
    trial = int(m.group(1))
    label = m.group(2)
    return trial, label


def make_preroll(
    out_path: Path,
    text: str,
    w: int,
    h: int,
    fps: float,
    n_frames: int,
    font_file: Optional[Path],
    font_size: int = 72,
    font_color: str = "black",
    bg_color: str = "white",
) -> None:
    duration = n_frames / fps
    drawtext_parts = []
    if font_file and font_file.exists():
        drawtext_parts.append(f"fontfile={font_file.as_posix()}")
    else:
        # Try named font via fontconfig as a fallback
        drawtext_parts.append("font=DejaVuSans-Bold")
    # Escape colons/backslashes in text for drawtext safety
    safe_text = text.replace("\\", "\\\\").replace(":", r"\:")
    drawtext_parts.extend(
        [
            f"text={safe_text}",
            f"fontcolor={font_color}",
            f"fontsize={font_size}",
            "box=1",
            "boxcolor=white@1.0",
            "boxborderw=20",
            "x=(w-text_w)/2",
            "y=(h-text_h)/2",
        ]
    )
    drawtext = ":".join(drawtext_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={bg_color}:s={w}x{h}:r={fps}:d={duration}",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf",
        f"drawtext={drawtext}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_path),
    ]
    run(cmd)


def quote_for_concat(p: Path) -> str:
    """Return a single-quoted absolute path for ffmpeg concat file list."""
    ap = p.resolve().as_posix()
    # Escape single quotes per ffmpeg concat demuxer rules (rare)
    ap = ap.replace("'", r"'\''")
    return f"file '{ap}'"


def main():
    ap = argparse.ArgumentParser(
        description="Concat clips with a trial flash preroll per clip."
    )
    ap.add_argument(
        "--clips-dir",
        required=True,
        type=Path,
        help="Directory containing per-trial clips.",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output mp4 path for concatenated video.",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=45,
        help="Number of preroll frames to show before each clip (default 45).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="trial_*.mp4",
        help="Glob pattern for clip filenames (default trial_*.mp4).",
    )
    ap.add_argument(
        "--font-file",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        help="Path to a TTF font file for drawtext.",
    )
    ap.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary preroll files."
    )
    ap.add_argument("--font-size", type=int, default=72, help="Font size for drawtext.")
    ap.add_argument(
        "--bg",
        type=str,
        default="white",
        help="Background color for preroll (default white).",
    )
    ap.add_argument(
        "--fontcolor", type=str, default="black", help="Font color (default black)."
    )
    ap.add_argument(
        "--debug", action="store_true", help="Print temp paths and concat list entries."
    )
    args = ap.parse_args()

    clips = find_clips(args.clips_dir, args.pattern)
    if not clips:
        print(f"No clips matched {args.pattern} in {args.clips_dir}", file=sys.stderr)
        sys.exit(2)

    # Determine standard params from first clip
    w, h, fps = get_video_params(clips[0])

    tmp_dir = args.output.parent / "__tmp_concat__"
    pre_dir = tmp_dir / "preroll"
    list_file = tmp_dir / "concat_list.txt"
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)

    list_lines = []
    items = []
    for clip in clips:
        trial, label = parse_trial_and_label(clip.name)
        text = f"TRIAL {trial:03d}"
        if label:
            text += f" — {label}"
        preroll_path = (pre_dir / f"{clip.stem}_preroll.mp4").resolve()
        make_preroll(
            out_path=preroll_path,
            text=text,
            w=w,
            h=h,
            fps=fps,
            n_frames=args.frames,
            font_file=args.font_file,
            font_size=args.font_size,
            font_color=args.fontcolor,
            bg_color=args.bg,
        )
        list_lines.append(quote_for_concat(preroll_path))
        list_lines.append(quote_for_concat(clip.resolve()))
        items.append(
            {
                "clip": clip.resolve().as_posix(),
                "trial": trial,
                "label": label,
                "preroll": preroll_path.as_posix(),
            }
        )

    list_file.write_text("\n".join(list_lines), encoding="utf-8")

    if args.debug:
        print(f"[DEBUG] temp dir: {tmp_dir.resolve()}")
        print(f"[DEBUG] list file: {list_file.resolve()}")
        print("[DEBUG] first 6 list lines:")
        for line in list_lines[:6]:
            print(" ", line)

    # Final concat with re-encode to normalize
    args.output.parent.mkdir(parents=True, exist_ok=True)
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-r",
        f"{fps:.6f}",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    run(concat_cmd)

    # Log provenance
    log_line = {
        "tool": "combine_clips_with_flash.py",
        "version": "0.1.1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clips_dir": str(args.clips_dir.resolve()),
        "pattern": args.pattern,
        "n_clips": len(clips),
        "fps_target": fps,
        "preroll_frames": args.frames,
        "output": str(args.output.resolve()),
        "items": items,
    }
    with open(log_dir / "concat.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line) + "\n")

    if not args.keep_temp:
        # clean up temp directory
        try:
            for p in pre_dir.glob("*.mp4"):
                p.unlink(missing_ok=True)
            list_file.unlink(missing_ok=True)
            pre_dir.rmdir()
            tmp_dir.rmdir()
        except Exception:
            # Non-fatal if cleanup fails
            pass

    print(f"Done. Wrote: {args.output}")
    print(f"Log appended to: {log_dir / 'concat.jsonl'}")


if __name__ == "__main__":
    main()
