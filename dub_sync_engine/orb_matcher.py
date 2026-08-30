"""
Tier 2 Fallback: Scale- and Aspect-Ratio Invariant ORB Feature Keypoint Matcher.
Matches cartoon and anime frames across differing aspect ratios (4:3 vs 16:9),
squished anamorphic geometry, heavy compression blur, and letterboxing.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from .visual_anchors import VisualAnchor, AnchorMatch
from .config import DubSyncConfig


class ORBMatcherEngine:
    """Scale- and geometry-invariant feature keypoint matcher using OpenCV ORB and RANSAC."""

    def __init__(self, config: DubSyncConfig):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_orb_features(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts Canny edge-enhanced ORB keypoints and descriptors."""
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhance line-art with Canny edges to focus on invariant cartoon artwork
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)

        keypoints, descriptors = self.orb.detectAndCompute(combined, None)
        return keypoints, descriptors

    def match_frame_pair(
        self,
        ref_kp, ref_desc,
        tar_kp, tar_desc,
        min_inliers: int = 12
    ) -> Tuple[bool, int, float]:
        """
        Matches ORB descriptors using Lowe's ratio test and Homography RANSAC.
        Returns (is_match, inlier_count, confidence).
        """
        if ref_desc is None or tar_desc is None or len(ref_desc) < 10 or len(tar_desc) < 10:
            return False, 0, 0.0

        # KNN matching with k=2
        matches = self.bf.knnMatch(ref_desc, tar_desc, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < min_inliers:
            return False, len(good_matches), 0.0

        # Geometric verification via Homography RANSAC
        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([tar_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return False, 0, 0.0

        inliers = int(np.sum(mask))
        if inliers >= min_inliers:
            conf = min(1.0, inliers / 40.0)
            return True, inliers, conf

        return False, inliers, 0.0

    def match_anchors_orb(
        self,
        ref_anchors: List[VisualAnchor],
        tar_anchors: List[VisualAnchor],
        progress_callback: Optional[callable] = None
    ) -> List[AnchorMatch]:
        """
        Extracts ORB descriptors on keyframes and finds scale-invariant anchor matches.
        """
        if not ref_anchors or not tar_anchors:
            return []

        # Pre-extract ORB features for all anchors
        ref_feats = []
        for r in ref_anchors:
            kp, desc = self.extract_orb_features(r.image_path)
            ref_feats.append((kp, desc))

        tar_feats = []
        for t in tar_anchors:
            kp, desc = self.extract_orb_features(t.image_path)
            tar_feats.append((kp, desc))

        candidates = []
        for r_idx, r in enumerate(ref_anchors):
            r_kp, r_desc = ref_feats[r_idx]
            if r_desc is None:
                continue

            for t_idx, t in enumerate(tar_anchors):
                t_kp, t_desc = tar_feats[t_idx]
                if t_desc is None:
                    continue

                is_match, inliers, conf = self.match_frame_pair(r_kp, r_desc, t_kp, t_desc)
                if is_match:
                    candidates.append({
                        "r_idx": r_idx,
                        "t_idx": t_idx,
                        "ref_time": r.pts_time,
                        "tar_time": t.pts_time,
                        "inliers": inliers,
                        "confidence": conf,
                        "offset": t.pts_time - r.pts_time,
                        "score": inliers * 1.0
                    })

            if progress_callback:
                progress_callback(r_idx + 1, len(ref_anchors))

        if not candidates:
            return []

        # Sort and solve monotonic path via Dynamic Programming
        candidates.sort(key=lambda x: (x["ref_time"], x["tar_time"]))

        N = len(candidates)
        dp = [c["score"] for c in candidates]
        parent = [-1] * N

        for i in range(N):
            ci = candidates[i]
            for j in range(i):
                cj = candidates[j]
                if cj["ref_time"] < ci["ref_time"] and cj["tar_time"] < ci["tar_time"]:
                    dt_r = ci["ref_time"] - cj["ref_time"]
                    dt_t = ci["tar_time"] - cj["tar_time"]
                    if dt_r <= 0:
                        continue
                    speed = dt_t / dt_r
                    if 0.88 <= speed <= 1.12:
                        gain = ci["score"] - abs(speed - 1.0) * 10.0
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j
                    elif dt_r > 4.0:
                        gain = ci["score"] - 2.0
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j

        best_idx = int(np.argmax(dp))
        chain = []
        curr = best_idx
        while curr != -1:
            c = candidates[curr]
            chain.append(AnchorMatch(
                ref_idx=c["r_idx"],
                tar_idx=c["t_idx"],
                ref_time=c["ref_time"],
                tar_time=c["tar_time"],
                hash_dist=max(0, 10 - int(c["inliers"] / 3)),
                confidence=round(c["confidence"], 3),
                offset=round(c["offset"], 4),
                source="orb"
            ))
            curr = parent[curr]

        chain.reverse()
        return chain
