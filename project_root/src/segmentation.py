import cv2
import numpy as np
from typing import Optional


def compute_background_cap(
    video_path: str, method: str, n: int
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    for _ in range(max(1, int(n))):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        return None
    if method == "median":
        return np.median(np.stack(frames, 0), 0).astype(np.uint8)
    return frames[0]  # fallback: first frame


def apply_roi(mask: np.ndarray, roi_mask: Optional[np.ndarray], erosion_px: int = 0):
    if roi_mask is None:
        return mask
    rm = roi_mask.astype(np.uint8)
    if erosion_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1)
        )
        rm = cv2.erode(rm, k)
    return cv2.bitwise_and(mask, mask, mask=rm)


def _largest_component(
    bw: np.ndarray, min_area: int, max_area: int, keep_largest: bool = True
):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return np.zeros_like(bw)
    areas = stats[:, cv2.CC_STAT_AREA]
    candidates = [
        i for i in range(1, num) if (areas[i] >= min_area and areas[i] <= max_area)
    ]
    if not candidates:
        return np.zeros_like(bw)
    if keep_largest:
        i = max(candidates, key=lambda idx: areas[idx])
        return (labels == i).astype(np.uint8) * 255
    out = np.zeros_like(bw)
    for i in candidates:
        out[labels == i] = 255
    return out


def segment_frame(
    gray: np.ndarray,
    bg: Optional[np.ndarray],
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
):
    g = gray
    fg = cv2.absdiff(g, bg) if bg is not None else g

    seg_cfg = cfg.get("seg", {})
    thr_mode = str(seg_cfg.get("threshold", "otsu")).lower()
    if thr_mode == "adaptive":
        blk = int(seg_cfg.get("adaptive_block", 41)) | 1
        C = int(seg_cfg.get("adaptive_C", 3))
        bw = cv2.adaptiveThreshold(
            fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, C
        )
        bw = 255 - bw
    else:
        _, bw = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphology
    open_it = int(seg_cfg.get("morph_open", 0))
    close_it = int(seg_cfg.get("morph_close", 0))
    if open_it > 0:
        k = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=open_it)
    if close_it > 0:
        k = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_it)

    # area + largest component
    min_area = int(seg_cfg.get("min_area", 0))
    max_area = int(seg_cfg.get("max_area", 10**9))
    keep_largest = bool(seg_cfg.get("keep_largest", True))
    mask = _largest_component(bw, min_area, max_area, keep_largest)

    # ROI last
    roi_cfg = cfg.get("roi", {})
    erosion_px = int(roi_cfg.get("erosion_px", 0))
    mask = apply_roi(mask, roi_mask, erosion_px)
    return mask
