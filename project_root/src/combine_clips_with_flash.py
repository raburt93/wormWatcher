#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine per-trial clips, inserting a preroll "flash" card before each clip.
Robust A/V sync via:
  - Sample-accurate preroll beep (optional)
  - Final concat via the *concat filter* with correct [v][a] pairing
  - Auto-silence generation for clips missing audio tracks

Changelog:
  v0.3.1
   - FIX: Concat filter now orders inputs as [v0][a0][v1][a1]... (correct).
   - NEW: Clips with no audio get a generated silent track of matching length.
   - KEEP: Sample-accurate preroll beep and normalization.

Usage (example):
  python combine_clips_with_flash.py \
      --clips-dir out/clips/D5W1P28Jun25 \
      --output out/qc/D5W1P28Jun25_with_titles.mp4 \
      --frames 45 \
      --pattern "trial_*.mp4" \
      --font-file "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" \
      --debug

Optional audio params:
  --beep-freq 1000         # Hz
  --beep-gain-db -18       # dBFS (negative = quieter)
  --no-beep                # silence on preroll instead of beep
  --sr 48000               # target audio sample rate (default 48000)

Requirements:
  - Python 3.10+
  - ffmpeg with drawtext, sine, aac, libx264 on PATH

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


def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("\n[ffmpeg output]\n" + p.stdout, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(shlex.quote(c) for c in cmd)}")
    return p.stdout


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
    return json.loads(run(cmd))


def parse_fps(rate: str) -> float:
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


def get_duration_seconds(path: Path) -> float:
    meta = ffprobe_json(path)
    dur = meta.get("format", {}).get("duration")
    try:
        return float(dur)
    except (TypeError, ValueError):
        # Fallback: estimate from video stream if nb_frames present
        for s in meta.get("streams", []):
            if (
                s.get("codec_type") == "video"
                and s.get("nb_frames")
                and s.get("avg_frame_rate")
            ):
                try:
                    n = float(s["nb_frames"])
                    r = parse_fps(s["avg_frame_rate"])
                    return n / r
                except Exception:
                    pass
        return 0.0


def has_audio_stream(path: Path) -> bool:
    meta = ffprobe_json(path)
    return any(s.get("codec_type") == "audio" for s in meta.get("streams", []))


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_clips(clips_dir: Path, pattern: str) -> List[Path]:
    files = sorted(clips_dir.glob(pattern), key=lambda p: natural_key(p.name))
    return [p.resolve() for p in files if p.is_file()]


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
    beep_freq: int = 1000,
    beep_gain_db: float = -18.0,
    beep_enabled: bool = True,
    sr: int = 48000,
) -> None:
    """
    Build a preroll MP4 with EXACT n_frames of video and audio trimmed to exactly
    round(n_frames / fps * sr) samples.
    """
    duration = n_frames / fps
    samples = int(round(duration * sr))

    # Prepare drawtext; escape backslashes and colons
    drawtext_parts = []
    if font_file and font_file.exists():
        drawtext_parts.append(f"fontfile={font_file.as_posix()}")
    else:
        drawtext_parts.append("font=DejaVuSans-Bold")
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

    # Audio source: sine (quiet beep) or anullsrc, both trimmed to exact samples
    if beep_enabled:
        audio_src = f"sine=frequency={beep_freq}:sample_rate={sr}"
        agraph = f"{audio_src},volume={beep_gain_db}dB,atrim=end_sample={samples},asetpts=N/SR/TB"
    else:
        audio_src = f"anullsrc=channel_layout=stereo:sample_rate={sr}"
        agraph = f"{audio_src},atrim=end_sample={samples},asetpts=N/SR/TB"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={bg_color}:s={w}x{h}:r={fps}",
        "-f",
        "lavfi",
        "-i",
        agraph,
        "-vf",
        f"drawtext={drawtext}",
        "-frames:v",
        str(n_frames),  # GUARANTEE exactly n preroll frames
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(sr),
        "-b:a",
        "128k",
        str(out_path),
    ]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(
        description="Concat clips with a trial flash preroll per clip (concat *filter* with sample-accurate beep)."
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
        "--beep-freq",
        type=int,
        default=1000,
        help="Beep frequency in Hz (default 1000).",
    )
    ap.add_argument(
        "--beep-gain-db",
        type=float,
        default=-18.0,
        help="Beep volume in dBFS (negative is quieter; default -18).",
    )
    ap.add_argument(
        "--no-beep", action="store_true", help="Disable beep; preroll will be silent."
    )
    ap.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Target audio sample rate for output and prerolls (default 48000).",
    )
    ap.add_argument(
        "--debug", action="store_true", help="Print filtergraph and temp paths."
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
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)

    # Build input list and per-segment (v_idx, a_idx) mapping
    ff_args = ["ffmpeg", "-y"]
    seg_map: List[Dict[str, int]] = []  # [{v: idx, a: idx}, ...], one per segment
    items = []

    def add_file_input(pth: Path) -> int:
        ff_args.extend(["-i", str(pth)])
        return (
            int((len(ff_args) - 1) / 2) - 1
        )  # rough index calc not reliable; instead compute from count

    # Instead, track explicitly
    input_specs: List[
        Tuple[str, str]
    ] = []  # list of ("file", path) or ("lavfi", lavfi_desc_with_t)

    def push_file(path_str: str) -> int:
        input_specs.append(("file", path_str))
        return len(input_specs) - 1

    def push_lavfi(desc: str, t: Optional[float] = None) -> int:
        if t is not None:
            input_specs.append(("lavfi", f"-t {t:.6f} -i {desc}"))
        else:
            input_specs.append(("lavfi", f"-i {desc}"))
        return len(input_specs) - 1

    # Prepare segments: [preroll, clip, preroll, clip, ...]
    for clip in clips:
        trial, label = parse_trial_and_label(clip.name)
        text = f"TRIAL {trial:03d}"
        if label:
            text += f" — {label}"
        # preroll
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
            beep_freq=args.beep_freq,
            beep_gain_db=args.beep_gain_db,
            beep_enabled=not args.no_beep,
            sr=args.sr,
        )
        p_idx = push_file(preroll_path.as_posix())
        seg_map.append({"v": p_idx, "a": p_idx})
        # clip
        c_v_idx = push_file(clip.as_posix())
        if has_audio_stream(clip):
            c_a_idx = c_v_idx
        else:
            dur = max(get_duration_seconds(clip), 0.001)
            # Generate silence matching the clip duration
            lavfi = f"anullsrc=channel_layout=stereo:sample_rate={args.sr}"
            c_a_idx = push_lavfi(lavfi, t=dur)
        seg_map.append({"v": c_v_idx, "a": c_a_idx})
        items.append(
            {
                "clip": clip.as_posix(),
                "trial": trial,
                "label": label,
                "preroll": preroll_path.as_posix(),
                "clip_has_audio": has_audio_stream(clip),
            }
        )

    # Now build ffmpeg input args
    ff_args = ["ffmpeg", "-y"]
    for kind, spec in input_specs:
        if kind == "file":
            ff_args += ["-i", spec]
        else:  # lavfi
            # spec already includes -t and -i parts
            ff_args += ["-f", "lavfi"]
            parts = spec.split()
            ff_args += parts  # "-t", "D", "-i", "anullsrc=..."
    n_inputs = len(input_specs)

    # Build filtergraph: normalize each stream, then concat in interleaved [v][a] pairs
    parts = []
    # Create normalized labels per input index
    for i in range(n_inputs):
        parts.append(
            f"[{i}:v]scale={w}:{h}:flags=bicubic,fps={fps},format=yuv420p,setpts=N/FRAME_RATE/TB[v{i}]"
        )
        parts.append(
            f"[{i}:a]aresample={args.sr}:resampler=soxr,aformat=sample_fmts=s16:channel_layouts=stereo,asetpts=N/SR/TB[a{i}]"
        )
    # Build interleaved list per segment
    cat_inputs = []
    for k, mp in enumerate(seg_map):
        cat_inputs.append(f"[v{mp['v']}]")
        cat_inputs.append(f"[a{mp['a']}]")
    parts.append("".join(cat_inputs) + f"concat=n={len(seg_map)}:v=1:a=1[vout][aout]")
    filtergraph = ";".join(parts)

    if args.debug:
        print("[DEBUG] Using filtergraph:")
        print(filtergraph)

    ff_args += [
        "-filter_complex",
        filtergraph,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(args.sr),
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(args.output),
    ]

    run(ff_args)

    # Log provenance
    log_line = {
        "tool": "combine_clips_with_flash.py",
        "version": "0.3.1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clips_dir": str(args.clips_dir.resolve()),
        "pattern": args.pattern,
        "n_clips": len(clips),
        "fps_target": fps,
        "preroll_frames": args.frames,
        "output": str(args.output.resolve()),
        "beep": None
        if args.no_beep
        else {"freq": args.beep_freq, "gain_db": args.beep_gain_db},
        "sr": args.sr,
        "items": items,
    }
    with open(log_dir / "concat.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line) + "\n")

    if not args.keep_temp:
        # clean up temp directory
        try:
            # keep tmp on failure only
            for p in pre_dir.glob("*.mp4"):
                p.unlink(missing_ok=True)
            pre_dir.rmdir()
            tmp_dir.rmdir()
        except Exception:
            pass

    print(f"Done. Wrote: {args.output}")
    print(f"Log appended to: {log_dir / 'concat.jsonl'}")


if __name__ == "__main__":
    main()
