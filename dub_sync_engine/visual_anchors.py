"""
Visual scene-change detection, safe-zone center crop, multi-descriptor feature extraction,
and Monotonic Dynamic Programming Lattice solver.
"""

import os
import re
import time
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import imagehash
import numpy as np
import cv2

from .media_probe import FFMPEG_PATH, run_ffmpeg_cmd
from .config import DubSyncConfig


@dataclass
class VisualAnchor:
    index: int
    pts_time: float
    image_path: str
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash
    color_hist: np.ndarray
    burst_hashes: List[imagehash.ImageHash] = field(default_factory=list)


@dataclass
class AnchorMatch:
    ref_idx: int
    tar_idx: int
    ref_time: float
    tar_time: float
    hash_dist: int
    confidence: float
    offset: float  # tar_time - ref_time
    seq_len: int = 1  # number of consecutive cut matches verified (N-gram rhythm)
    weight: float = 1.0  # continuous confirmation strength (acoustic × N-gram); feeds RANSAC fit
    source: str = "unknown"  # modality that produced this anchor (acoustic_music/vad_speech/visual_gated/visual/orb/spectral/vad)
    acoustic_shift_ms: float = 0.0  # sub-ms acoustic refinement applied to this anchor
    acoustic_confidence: float = 1.0  # normalized acoustic cross-correlation peak at refinement


class VisualAnchorEngine:
    """Extracts, hashes, and aligns visual scene transition anchors."""

    def __init__(self, config: DubSyncConfig):
        self.config = config

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: str,
        prefix: str,
        max_duration: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> List[VisualAnchor]:
        """
        Extracts keyframes at scene transitions with safe-zone center cropping
        and multi-descriptor visual features.
        """
        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()

        # Build FFmpeg filter: Crop central 80% (safe zone) -> Scale to 320x180 -> Scene detection
        crop_pct = self.config.center_crop_ratio
        crop_filter = f"crop=iw*{crop_pct}:ih*{crop_pct}:(iw-iw*{crop_pct})/2:(ih-ih*{crop_pct})/2"
        scale_filter = "scale=320:180"
        scene_filter = f"select='gt(scene,{self.config.scene_threshold})'"
        full_vf = f"{crop_filter},{scale_filter},{scene_filter},showinfo"

        cmd = [
            FFMPEG_PATH, "-hide_banner",
            "-threads", str(self.config.num_threads),
        ]
        if max_duration:
            cmd.extend(["-t", str(max_duration)])

        cmd.extend([
            "-i", video_path,
            "-vf", full_vf,
            "-fps_mode", "vfr",
            "-q:v", "3",
            os.path.join(output_dir, f"{prefix}_%05d.jpg"),
            "-y"
        ])

        proc = run_ffmpeg_cmd(cmd, desc=f"Extracting Keyframes: {os.path.basename(video_path)}")

        # Parse exact pts_time from showinfo stderr
        anchors = []
        parsed_times = []
        for line in proc.stderr.splitlines():
            if "pts_time:" in line:
                m = re.search(r"n:\s*(\d+)\s+pts:\s*(\d+)\s+pts_time:([0-9.]+)", line)
                if m:
                    n = int(m.group(1))
                    pts_time = float(m.group(3))
                    parsed_times.append((n, pts_time))

        for n, pts_time in parsed_times:
            img_file = os.path.join(output_dir, f"{prefix}_{n+1:05d}.jpg")
            if os.path.exists(img_file):
                try:
                    cv_img = cv2.imread(img_file)
                    if cv_img is None:
                        continue

                    # Auto-crop black border rows/cols (handles letterboxing & pillarboxing)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    mask = gray > 18
                    if np.any(mask):
                        y_nz, x_nz = np.nonzero(mask)
                        cropped = cv_img[np.min(y_nz):np.max(y_nz)+1, np.min(x_nz):np.max(x_nz)+1]
                        if cropped.shape[0] > 10 and cropped.shape[1] > 10:
                            cv_img = cropped

                    # Standardize to 16:9 normalized canvas
                    cv_img = cv2.resize(cv_img, (320, 180))

                    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    p_h = imagehash.phash(pil_img)
                    d_h = imagehash.dhash(pil_img)

                    # Compute 3D color histogram in HSV
                    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    anchor = VisualAnchor(
                        index=n,
                        pts_time=pts_time,
                        image_path=img_file,
                        phash=p_h,
                        dhash=d_h,
                        color_hist=hist
                    )
                    anchors.append(anchor)

                    if progress_callback:
                        progress_callback(len(anchors))

                except Exception:
                    pass

        return anchors

    def match_anchors(
        self,
        ref_anchors: List[VisualAnchor],
        tar_anchors: List[VisualAnchor]
    ) -> List[AnchorMatch]:
        """
        Solves optimal monotonic anchor matching via Dynamic Programming Lattice with
        Multi-Frame Temporal Sequence Consistency.
        Boosts sequences of consecutive camera cuts (F_k -> F_k+1) and penalizes isolated noise.
        """
        if not ref_anchors or not tar_anchors:
            return []

        # Step 1: Candidate generation using multi-descriptor composite distance
        raw_candidates = []
        for r_idx, r in enumerate(ref_anchors):
            for t_idx, t in enumerate(tar_anchors):
                # Combined distance: pHash (frequency) + dHash (gradient)
                p_dist = r.phash - t.phash
                d_dist = r.dhash - t.dhash
                composite_dist = (p_dist + d_dist) / 2.0

                if composite_dist <= self.config.max_hash_dist:
                    # Color histogram correlation
                    color_sim = cv2.compareHist(r.color_hist, t.color_hist, cv2.HISTCMP_CORREL)
                    if color_sim < 0.15:  # Reject if color distributions completely disagree
                        continue

                    # Base confidence and score
                    confidence = max(0.0, 1.0 - (composite_dist / 12.0)) * (0.5 + 0.5 * max(0.0, color_sim))
                    score = (12.0 - composite_dist) + (color_sim * 4.0)

                    # Step 1b: Multi-Frame Temporal Sequence Verification
                    # Look ahead 1 to 2 keyframes to verify sequence rhythm
                    seq_len = 1
                    for k in range(1, 3):
                        if (r_idx + k) < len(ref_anchors) and (t_idx + k) < len(tar_anchors):
                            rk = ref_anchors[r_idx + k]
                            tk = tar_anchors[t_idx + k]
                            p_k = rk.phash - tk.phash
                            d_k = rk.dhash - tk.dhash
                            c_k = (p_k + d_k) / 2.0
                            if c_k <= self.config.max_hash_dist + 2:
                                dr = rk.pts_time - r.pts_time
                                dt = tk.pts_time - t.pts_time
                                if dr > 0.1 and dt > 0.1:
                                    ratio = dt / dr
                                    if 0.92 <= ratio <= 1.05:
                                        seq_len += 1
                                        score += (10.0 - c_k) * 1.5

                    if seq_len >= 2:
                        confidence = min(1.0, confidence + 0.10)
                        score += 6.0 * (seq_len - 1)
                    if seq_len >= 3:
                        # A chain of 3 consecutive camera cuts is virtually unique in
                        # animation — grant a strong confidence boost.
                        confidence = min(1.0, confidence + 0.20)
                        score += 15.0 * (seq_len - 1)

                    raw_candidates.append({
                        "r_idx": r_idx,
                        "t_idx": t_idx,
                        "ref_time": r.pts_time,
                        "tar_time": t.pts_time,
                        "hash_dist": int(composite_dist),
                        "confidence": float(confidence),
                        "offset": t.pts_time - r.pts_time,
                        "score": float(score),
                        "seq_len": seq_len
                    })

        if not raw_candidates:
            return []

        # Step 2: Sort candidates by reference timestamp
        raw_candidates.sort(key=lambda x: (x["ref_time"], x["tar_time"]))

        # Step 3: Longest strictly monotonic increasing path with deformation penalty
        N = len(raw_candidates)
        dp = [c["score"] for c in raw_candidates]
        parent = [-1] * N

        for i in range(N):
            ci = raw_candidates[i]
            for j in range(i):
                cj = raw_candidates[j]

                # Must be strictly forward in both reference and target timelines
                if cj["ref_time"] < ci["ref_time"] and cj["tar_time"] < ci["tar_time"]:
                    dt_r = ci["ref_time"] - cj["ref_time"]
                    dt_t = ci["tar_time"] - cj["tar_time"]

                    if dt_r <= 0.1:
                        continue

                    speed_ratio = dt_t / dt_r

                    # Physics deformation penalty:
                    # Continuous scenes should have speed_ratio in broadcast tempo [0.92, 1.05].
                    if 0.92 <= speed_ratio <= 1.05:
                        speed_penalty = abs(speed_ratio - 0.960) * 4.0
                        gain = ci["score"] - speed_penalty
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j
                    elif dt_r >= 3.5 and speed_ratio < 0.92:  # Cut / scene omission bridge
                        cut_penalty = 2.0
                        gain = ci["score"] - cut_penalty
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j

        # Step 4: Backtrack optimal path
        best_idx = int(np.argmax(dp))
        chain = []
        curr = best_idx
        while curr != -1:
            c = raw_candidates[curr]
            match = AnchorMatch(
                ref_idx=c["r_idx"],
                tar_idx=c["t_idx"],
                ref_time=c["ref_time"],
                tar_time=c["tar_time"],
                hash_dist=c["hash_dist"],
                confidence=round(c["confidence"], 3),
                offset=round(c["offset"], 4),
                seq_len=c["seq_len"],
                source="visual"
            )
            chain.append(match)
            curr = parent[curr]

        chain.reverse()
        return chain
