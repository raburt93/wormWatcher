import numpy as np


class OrientationFilter:
    """Hybrid head/tail decision with motion vote, tip curvature, and clitellum bias."""

    def __init__(
        self, window_frames=5, flip_hysteresis=3, tip_curv_win=5, clitellum_bias=0.2
    ):
        self.win = int(window_frames)
        self.flip_hyst = int(flip_hysteresis)
        self.tip_win = int(tip_curv_win)
        self.bias = float(clitellum_bias)
        self.prev_head = None
        self.flip_count = 0

    def _curvature_tip(self, path):
        def curv_score(sub):
            if len(sub) < 3:
                return 0.0
            v = np.diff(np.array(sub), axis=0).astype(np.float32)
            n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
            v = v / n
            ang = np.arccos(np.clip((v[:-1] * v[1:]).sum(1), -1, 1))
            return float(np.sum(ang))

        k = min(self.tip_win, max(2, len(path) // 4))
        return curv_score(path[:k]), curv_score(path[-k:])

    def choose_head(self, path, prev_path, clit_idx):
        if not path:
            return None
        # motion vote
        if prev_path and len(prev_path) and len(path):
            a_now = np.array(path[0])
            b_now = np.array(path[-1])
            a_prev = np.array(prev_path[0])
            b_prev = np.array(prev_path[-1])
            ma = float(np.linalg.norm(a_now - a_prev))
            mb = float(np.linalg.norm(b_now - b_prev))
            motion_vote_head = 0 if ma >= mb else -1
        else:
            motion_vote_head = 0
        # tip sharpness vote (head sharper)
        curv_a, curv_b = self._curvature_tip(path)
        sharp_vote_head = 0 if curv_a >= curv_b else -1
        # clitellum proximity vote (tail closer to clitellum)
        if clit_idx is not None:
            dist_a = abs(0 - clit_idx)
            dist_b = abs((len(path) - 1) - clit_idx)
            clit_vote_head = 0 if dist_a > dist_b else -1
        else:
            clit_vote_head = 0

        # combine votes (clit vote down-weighted by bias)
        votes = [
            motion_vote_head,
            sharp_vote_head,
            (0 if clit_vote_head == 0 else -1) * self.bias,
        ]
        score = sum(1.0 if v == 0 else -1.0 for v in votes)
        new_head_idx = 0 if score >= 0 else -1

        # hysteresis to prevent flapping
        if self.prev_head is not None and new_head_idx != self.prev_head:
            self.flip_count += 1
            if self.flip_count < self.flip_hyst:
                new_head_idx = self.prev_head
            else:
                self.flip_count = 0
        self.prev_head = new_head_idx
        return new_head_idx
