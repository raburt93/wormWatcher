import numpy as np
import cv2
from scipy.signal import savgol_filter


def distance_profile(mask: np.ndarray, path):
    # Euclidean distance transform inside the worm region
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return np.array([dt[y, x] for (y, x) in path], dtype=np.float32)


def find_clitellum_index(
    radii: np.ndarray, search_range_pct=(0.25, 0.60), smooth_win=9, min_prom_px=2.5
):
    if len(radii) == 0:
        return None
    n = len(radii)
    # ensure odd window and <= n
    if smooth_win % 2 == 0:
        smooth_win += 1
    win = min(smooth_win, n if n % 2 == 1 else n - 1)
    if win < 3:
        win = 3
    r = savgol_filter(radii, window_length=win, polyorder=2, mode="interp")

    i0 = max(0, min(int(search_range_pct[0] * n), n - 1))
    i1 = max(i0 + 1, min(int(search_range_pct[1] * n), n))
    seg = r[i0:i1]
    if len(seg) == 0:
        return None

    j = int(np.argmax(seg))
    if seg[j] < float(min_prom_px):
        return None
    return i0 + j
