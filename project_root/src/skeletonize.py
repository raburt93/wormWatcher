import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis


def _bridge_small_gaps(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return closed


def endpoint_coords(skel: np.ndarray):
    Ys, Xs = np.where(skel > 0)
    pts = []
    for y, x in zip(Ys, Xs):
        nb = skel[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2]
        # sum counts pixels (0/1); endpoint ~ self + 1 neighbor => 2 (4- or 8-conn approx)
        if nb.sum() in (2, 3):
            pts.append((y, x))
    return pts


def _spur_prune(skel: np.ndarray, spur_px: int) -> np.ndarray:
    if spur_px <= 0:
        return skel
    s = skel.copy()
    # iteratively peel endpoints to shorten small spurs
    for _ in range(spur_px):
        eps = endpoint_coords(s)
        if not eps:
            break
        for y, x in eps:
            s[y, x] = 0
    return s


def order_longest_path(skel: np.ndarray):
    endpoints = endpoint_coords(skel)
    if len(endpoints) < 2:
        ys, xs = np.where(skel > 0)
        return list(zip(ys, xs))
    coords_set = set(zip(*np.where(skel > 0)))
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    best_path = []
    for s_idx in range(len(endpoints)):
        sy, sx = endpoints[s_idx]
        from collections import deque

        q = deque([(sy, sx)])
        prev = {(sy, sx): None}
        seen = {(sy, sx)}
        far = (sy, sx)
        while q:
            y, x = q.popleft()
            far = (y, x)
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if (ny, nx) in seen:
                    continue
                if (ny, nx) in coords_set:
                    seen.add((ny, nx))
                    prev[(ny, nx)] = (y, x)
                    q.append((ny, nx))
        # build path back
        path = []
        cur = far
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        if len(path) > len(best_path):
            best_path = path[::-1]
    return best_path


def make_skeleton(mask: np.ndarray, method="medial_axis", spur_px=6, bridge_gaps_px=2):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    bridged = _bridge_small_gaps(mask_bin, int(bridge_gaps_px))
    m = bridged > 0
    if method == "thin":
        sk = skeletonize(m).astype(np.uint8)
    else:
        sk = medial_axis(m).astype(np.uint8)
    sk = _spur_prune(sk, int(spur_px))
    path = order_longest_path(sk)
    return sk, path
