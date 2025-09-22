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


def _apply_border_kill(bw: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return bw
    H, W = bw.shape[:2]
    bw[:px, :] = 0
    bw[-px:, :] = 0
    bw[:, :px] = 0
    bw[:, -px:] = 0
    return bw


def _axis_ratio(stats_row) -> float:
    w = max(1, int(stats_row[cv2.CC_STAT_WIDTH]))
    h = max(1, int(stats_row[cv2.CC_STAT_HEIGHT]))
    return float(max(w, h) / max(1, min(w, h)))


def _thickness_stats(mask_u8: np.ndarray):
    dt = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dt[mask_u8 > 0]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    # thickness ~ 2*distance
    return float(np.median(vals) * 2), float(np.mean(vals) * 2), float(np.max(vals) * 2)


def _contrast_ring(gray: np.ndarray, comp_u8: np.ndarray, ring_px: int) -> float:
    if ring_px <= 0:
        # fallback: use global background
        inside = gray[comp_u8 > 0]
        return float(np.mean(gray) - np.mean(inside)) if inside.size else 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    dil = cv2.dilate(comp_u8, k)
    ring = cv2.subtract(dil, comp_u8)  # ring shell
    inside = gray[comp_u8 > 0]
    around = gray[ring > 0]
    if inside.size == 0 or around.size == 0:
        return 0.0
    return float(np.mean(around) - np.mean(inside))


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
    prev_gray: Optional[np.ndarray] = None,
):
    """
    Segment worm with polarity-aware thresholding, black-hat prefilter,
    and component selection gated by ROI overlap, border distance, axis ratio,
    thickness, contrast, and (optionally) motion.
    """
    seg_cfg = cfg.get("seg", {})
    roi_cfg = cfg.get("roi", {})
    mov_cfg = cfg.get("motion", {})
    pre_cfg = cfg.get("preproc", {})

    # ---- preproc: black-hat to enhance thin dark structures ----
    g = gray.copy()
    ksize = int(pre_cfg.get("blackhat_ksize", 0))
    if ksize and ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        g = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)

    # ---- background subtract if available ----
    fg = cv2.absdiff(g, bg) if bg is not None else g

    # ---- threshold with polarity ----
    fore = str(seg_cfg.get("foreground", "dark")).lower()  # dark|bright
    thr_mode = str(seg_cfg.get("threshold", "otsu")).lower()
    if thr_mode == "adaptive":
        blk = int(seg_cfg.get("adaptive_block", 41)) | 1
        C = int(seg_cfg.get("adaptive_C", 3))
        # THRESH_BINARY makes bright=255; invert if we want dark foreground
        flag = cv2.THRESH_BINARY if fore == "bright" else cv2.THRESH_BINARY_INV
        bw = cv2.adaptiveThreshold(
            fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, blk, C
        )
    else:
        # Otsu with polarity
        flag = cv2.THRESH_BINARY if fore == "bright" else cv2.THRESH_BINARY_INV
        _, bw = cv2.threshold(fg, 0, 255, flag + cv2.THRESH_OTSU)

    # ---- morphology ----
    open_it = int(seg_cfg.get("morph_open", 0))
    close_it = int(seg_cfg.get("morph_close", 0))
    if open_it > 0:
        bw = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=open_it
        )
    if close_it > 0:
        bw = cv2.morphologyEx(
            bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_it
        )

    H, W = bw.shape[:2]
    border_kill = int(roi_cfg.get("border_kill_px", 0))
    bw = _apply_border_kill(bw, border_kill)

    # If ROI provided, hard-clip to eroded ROI to avoid edge effects
    if roi_mask is not None:
        er_px = int(roi_cfg.get("erosion_px", 0)) + border_kill
        if er_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (er_px * 2 + 1, er_px * 2 + 1)
            )
            roi_er = cv2.erode(roi_mask.astype(np.uint8), k)
        else:
            roi_er = roi_mask.astype(np.uint8)
        bw = cv2.bitwise_and(bw, bw, mask=roi_er)

    # ---- components ----
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return np.zeros_like(bw)

    # motion map (optional)
    moving = None
    if mov_cfg.get("use_temporal_gate", False) and prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        moving = (diff > int(mov_cfg.get("diff_thresh", 12))).astype(np.uint8)
    min_moving_frac = float(mov_cfg.get("min_moving_frac", 0.0))

    min_area = int(seg_cfg.get("min_area", 0))
    max_area = int(seg_cfg.get("max_area", 10**9))
    min_overlap_frac = float(roi_cfg.get("min_overlap_frac", 0.0))
    anchor_range = roi_cfg.get("anchor_x_range", None)
    anchor_boost = float(roi_cfg.get("anchor_boost", 0.0))

    geom = seg_cfg.get("geom", {}) if isinstance(seg_cfg.get("geom"), dict) else {}
    min_ar = float(geom.get("min_axis_ratio", 1.0))
    max_ar = float(geom.get("max_axis_ratio", 1e6))

    thick = (
        seg_cfg.get("thickness", {})
        if isinstance(seg_cfg.get("thickness"), dict)
        else {}
    )
    t_min = float(thick.get("min_px", 0.0))
    t_max = float(thick.get("max_px", 1e9))

    contr = (
        seg_cfg.get("contrast", {}) if isinstance(seg_cfg.get("contrast"), dict) else {}
    )
    ring_px = int(contr.get("ring_px", 8))
    min_delta = float(contr.get("min_delta", 0.0))

    def in_anchor(cx_norm: float) -> bool:
        if not anchor_range:
            return False
        a0, a1 = float(anchor_range[0]), float(anchor_range[1])
        return (cx_norm >= a0) and (cx_norm <= a1)

    candidates = []
    for i in range(1, num):
        st = stats[i]
        area = int(st[cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        # reject if bbox hugs the frame (likely tray/corner)
        lo = int(st[cv2.CC_STAT_LEFT])
        t = int(st[cv2.CC_STAT_TOP])
        r = lo + int(st[cv2.CC_STAT_WIDTH])
        b = t + int(st[cv2.CC_STAT_HEIGHT])
        if (
            lo <= border_kill
            or t <= border_kill
            or r >= W - border_kill
            or b >= H - border_kill
        ):
            continue

        comp_u8 = (labels == i).astype(np.uint8) * 255

        # ROI overlap
        if roi_mask is not None and min_overlap_frac > 0.0:
            overlap = (comp_u8.astype(bool) & roi_mask.astype(bool)).sum()
            if (overlap / max(1, area)) < min_overlap_frac:
                continue

        # axis ratio (elongation)
        ar = _axis_ratio(st)
        if not (min_ar <= ar <= max_ar):
            continue

        # thickness gate
        t_med, t_mean, _ = _thickness_stats(comp_u8)
        if not (t_min <= t_med <= t_max):
            continue

        # contrast gate: worm is darker than surroundings → positive delta
        delta = _contrast_ring(gray, comp_u8, ring_px)
        if delta < min_delta:
            continue
        if fore == "dark" and delta <= 0:
            continue
        if fore == "bright" and delta >= 0:
            continue

        # motion gate
        if moving is not None:
            mov_overlap = (moving.astype(bool) & (comp_u8.astype(bool))).sum()
            if (mov_overlap / max(1, area)) < min_moving_frac:
                continue

        # score: area + contrast bonus + anchor bonus
        cx, _ = cents[i]
        cx_norm = float(cx) / max(1.0, float(W) - 1.0)
        score = float(area) + 500.0 * max(0.0, delta)
        if in_anchor(cx_norm):
            score *= 1.0 + anchor_boost

        candidates.append((score, i))

    if not candidates:
        return np.zeros_like(bw)

    # keep best (or union if you prefer)
    mask = np.zeros_like(bw)
    _, idx = max(candidates, key=lambda t: t[0])
    mask[labels == idx] = 255
    return mask
