"""
Neural ML Voice Activity Detection (VAD) Engine using Silero VAD (ONNX Runtime).
Extracts speech probability curves and aligns dialogue bursts and pauses across languages,
completely bypassing speech vocabulary differences and background music interference.
"""

import os
import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
import onnxruntime
from typing import List, Dict, Tuple, Optional

from .visual_anchors import AnchorMatch
from .config import DubSyncConfig


class SileroVADEngine:
    """Neural Voice Activity Detector and dialogue burst alignment engine."""

    def __init__(self, config: Optional[DubSyncConfig] = None, model_path: Optional[str] = None):
        self.config = config or DubSyncConfig()
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "silero_vad.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Silero VAD ONNX model not found at {model_path}")

        # Configure lightweight ONNX session for pure CPU inference
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = onnxruntime.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.sample_rate = 16000
        self.chunk_size = 512  # 32ms at 16kHz

    def compute_speech_probabilities(self, wav_path: str) -> Tuple[np.ndarray, float]:
        """
        Runs streaming Silero VAD on uncompressed audio and returns continuous speech
        probability array P(t) along with the time step (in seconds) between samples.
        """
        sr, audio = wavfile.read(wav_path)

        # Convert to mono float32
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=1).astype(np.float32)
        else:
            audio_mono = audio.astype(np.float32)

        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(audio_mono))
        if max_val > 1.0:
            audio_mono = audio_mono / 32768.0

        # Resample to 16,000Hz if needed
        if sr == 48000 and self.sample_rate == 16000:
            audio_16k = audio_mono[::3].copy()
        elif sr != self.sample_rate:
            audio_16k = scipy.signal.resample_poly(audio_mono, self.sample_rate // 1000, sr // 1000).astype(np.float32)
        else:
            audio_16k = audio_mono

        # Ensure length is divisible by chunk_size
        pad_size = (self.chunk_size - (len(audio_16k) % self.chunk_size)) % self.chunk_size
        if pad_size > 0:
            audio_16k = np.pad(audio_16k, (0, pad_size), mode="constant")

        num_chunks = len(audio_16k) // self.chunk_size
        probabilities = np.zeros(num_chunks, dtype=np.float32)

        # Initialize recurrent state and 64-sample rolling context
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr_tensor = np.array(self.sample_rate, dtype=np.int64)
        context = np.zeros(64, dtype=np.float32)

        for i in range(num_chunks):
            chunk = audio_16k[i * self.chunk_size : (i + 1) * self.chunk_size]
            model_input = np.concatenate([context, chunk]).reshape(1, 576).astype(np.float32)

            inputs = {
                "input": model_input,
                "state": state,
                "sr": sr_tensor
            }
            out, state = self.session.run(None, inputs)
            probabilities[i] = float(out[0][0])
            context = chunk[-64:]

        time_step = self.chunk_size / float(self.sample_rate)  # 0.032s (32ms)
        return probabilities, time_step

    def discover_speech_anchors(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float,
        step_seconds: float = 25.0,
        probe_duration: float = 20.0
    ) -> List[AnchorMatch]:
        """
        Extracts speech probability envelopes and computes cross-correlation of dialogue
        bursts/pauses to discover robust speech anchors.
        """
        p_ref, dt_r = self.compute_speech_probabilities(ref_wav_path)
        p_tar, dt_t = self.compute_speech_probabilities(tar_wav_path)

        # Smooth probability curves slightly
        smooth_kernel = np.ones(3) / 3.0
        p_ref_s = np.convolve(p_ref, smooth_kernel, mode="same")
        p_tar_s = np.convolve(p_tar, smooth_kernel, mode="same")

        p_ref_s -= np.mean(p_ref_s)
        p_tar_s -= np.mean(p_tar_s)

        win_frames = int(probe_duration / dt_r)
        step_frames = int(step_seconds / dt_r)

        candidates = []
        for f_ref in range(int(10.0 / dt_r), len(p_ref_s) - win_frames - int(10.0 / dt_r), step_frames):
            t_ref = f_ref * dt_r
            ref_slice = p_ref_s[f_ref : f_ref + win_frames]
            norm_r = np.linalg.norm(ref_slice)

            # Skip if this slice has almost zero speech variance (pure silence throughout)
            if norm_r < 0.1:
                continue

            corr = scipy.signal.correlate(p_tar_s, ref_slice, mode="valid")
            best_lag = int(np.argmax(corr))
            tar_slice = p_tar_s[best_lag : best_lag + win_frames]
            norm_t = np.linalg.norm(tar_slice)

            norm_peak = float(corr[best_lag] / (norm_r * norm_t + 1e-8))

            if norm_peak >= 0.20:
                t_tar = best_lag * dt_t
                offset = t_tar - t_ref
                candidates.append({
                    "ref_time": float(t_ref),
                    "tar_time": float(t_tar),
                    "offset": float(offset),
                    "confidence": min(1.0, norm_peak * 1.5),
                    "score": norm_peak * 10.0
                })

        if not candidates:
            return []

        # Solve monotonic path via Dynamic Programming Lattice
        candidates.sort(key=lambda x: (x["ref_time"], x["tar_time"]))
        N = len(candidates)
        dp = [c["score"] for c in candidates]
        parent = [-1] * N

        for i in range(N):
            ci = candidates[i]
            for j in range(i):
                cj = candidates[j]
                if cj["ref_time"] < ci["ref_time"] and cj["tar_time"] < ci["tar_time"]:
                    dt_r_span = ci["ref_time"] - cj["ref_time"]
                    dt_t_span = ci["tar_time"] - cj["tar_time"]
                    if dt_r_span <= 0:
                        continue
                    speed = dt_t_span / dt_r_span
                    if 0.92 <= speed <= 1.08:
                        gain = ci["score"] - abs(speed - 1.0) * 8.0
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j
                    elif dt_r_span > 5.0:
                        gain = ci["score"] - 1.5
                        if dp[j] + gain > dp[i]:
                            dp[i] = dp[j] + gain
                            parent[i] = j

        best_idx = int(np.argmax(dp))
        chain = []
        curr = best_idx
        idx = 0
        while curr != -1:
            c = candidates[curr]
            chain.append(AnchorMatch(
                ref_idx=idx,
                tar_idx=idx,
                ref_time=round(c["ref_time"], 3),
                tar_time=round(c["tar_time"], 3),
                hash_dist=0,
                confidence=round(c["confidence"], 3),
                offset=round(c["offset"], 4)
            ))
            curr = parent[curr]
            idx += 1

        chain.reverse()
        return chain

    def compute_neural_dtw_edl(
        self,
        ref_wav_path: str,
        tar_wav_path: str,
        ref_duration: float,
        tar_duration: float
    ):
        """
        Computes a dense neural Dynamic Time Warping (DTW) alignment path between
        reference and foreign speech probability envelopes, and builds a frame-accurate EDL.
        """
        from .audio_splicer import SegmentEDL

        p_ref, dt_r = self.compute_speech_probabilities(ref_wav_path)
        p_tar, dt_t = self.compute_speech_probabilities(tar_wav_path)

        # Downsample probabilities by 4 (to ~128ms resolution for fast full-episode global DTW)
        p_r_4 = p_ref[::4]
        p_t_4 = p_tar[::4]
        dt_4 = dt_r * 4.0

        N_r = len(p_r_4)
        N_t = len(p_t_4)

        if N_r == 0 or N_t == 0:
            return [
                SegmentEDL(
                    seg_id=0,
                    segment_type="dub",
                    ref_start=0.0,
                    ref_end=ref_duration,
                    tar_start=0.0,
                    tar_end=tar_duration,
                    speed_factor=1.0,
                    confidence=0.5
                )
            ]

        band_frames = int(self.config.dtw_band_sec / dt_4)
        D = np.full((N_r + 1, N_t + 1), np.inf, dtype=np.float32)
        D[0, 0] = 0.0

        for i in range(1, N_r + 1):
            exp_j = int(i * (N_t / N_r))
            j_min = max(1, exp_j - band_frames)
            j_max = min(N_t + 1, exp_j + band_frames)
            val_r = p_r_4[i - 1]
            for j in range(j_min, j_max):
                cost = (val_r - p_t_4[j - 1]) ** 2
                D[i, j] = cost + min(D[i - 1, j - 1], D[i - 1, j] + 0.04, D[i, j - 1] + 0.04)

        path = []
        i, j = N_r, N_t
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            steps = [D[i - 1, j - 1], D[i - 1, j] + 0.04, D[i, j - 1] + 0.04]
            best = np.argmin(steps)
            if best == 0:
                i -= 1
                j -= 1
            elif best == 1:
                i -= 1
            else:
                j -= 1
        path.reverse()

        # Sample warping nodes (every ~20s)
        macro_nodes = []
        step = max(1, int(self.config.dtw_node_interval_sec / dt_4))
        for idx in range(0, len(path), step):
            f_r, f_t = path[idx]
            macro_nodes.append((f_r * dt_4, f_t * dt_4))

        last_node = (path[-1][0] * dt_4, path[-1][1] * dt_4)
        if macro_nodes[-1] != last_node:
            macro_nodes.append(last_node)

        # Build EDL from macro nodes
        edl = []
        seg_id = 0
        standards = [1.000000, 24.0 / 25.0, 25.0 / 24.0, 24.0 / 23.976, 23.976 / 24.0]

        for k in range(len(macro_nodes) - 1):
            t_r1, t_t1 = macro_nodes[k]
            t_r2, t_t2 = macro_nodes[k + 1]

            r_dur = t_r2 - t_r1
            t_dur = t_t2 - t_t1

            if r_dur <= 0.05:
                continue

            raw_speed = t_dur / r_dur
            speed = raw_speed
            for std in standards:
                if abs(raw_speed - std) < 0.005:
                    speed = std
                    break
            speed = max(0.90, min(1.10, speed))

            edl.append(SegmentEDL(
                seg_id=seg_id,
                segment_type="dub",
                ref_start=round(t_r1, 3),
                ref_end=round(t_r2, 3),
                tar_start=round(t_t1, 3),
                tar_end=round(t_t1 + r_dur * speed, 3),
                speed_factor=round(speed, 6),
                confidence=0.95
            ))
            seg_id += 1

        return edl
