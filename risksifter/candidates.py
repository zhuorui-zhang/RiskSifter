"""Deterministic, query-free proposal baseline; scores are NOT risk probabilities."""
import cv2
import numpy as np

STRATEGY_VERSION = "motion-residual-v1"


def scan(proxy, sample_fps=2.0):
    cap = cv2.VideoCapture(str(proxy))
    fps = cap.get(cv2.CAP_PROP_FPS)
    samples, previous = [], None
    n, next_sample = 0, 0.0
    cv2.setNumThreads(2)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / fps
            n += 1
            if t + 1e-6 < next_sample:
                continue
            next_sample += 1 / sample_fps
            gray = cv2.cvtColor(cv2.resize(frame, (160, 96)), cv2.COLOR_BGR2GRAY)
            # Exclude sky / dashboard and limit edge effects from ego-motion.
            gray = gray[24:86, 16:144]
            change, residual = 0.0, 0.0
            if previous is not None:
                change = float(np.mean(cv2.absdiff(gray, previous))) / 255
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 2, 13, 2, 5, 1.1, 0)
                center = np.median(flow.reshape(-1, 2), axis=0)
                residual = float(np.quantile(np.linalg.norm(flow - center, axis=2), .9)) / 128
            score = .4 * change + .6 * min(residual, 1.0)
            samples.append({"time": round(t, 6), "score": round(score, 6),
                            "appearance_change": round(change, 6), "residual_flow": round(residual, 6)})
            previous = gray
    finally:
        cap.release()
    if not samples:
        raise ValueError("Video could not be sampled")
    return samples


def tiou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


class MotionStrategy:
    def propose(self, samples, duration, config):
        window = min(duration, config.window_seconds)
        if config.strategy == "uniform":
            centers = np.linspace(window / 2, duration - window / 2, config.max_candidates)
            ranked = [{"time": float(t), "score": 0.0} for t in centers]
        else:
            ranked = sorted(samples, key=lambda s: (-s["score"], s["time"]))
        selected = []
        for sample in ranked:
            if config.strategy == "motion" and sample["score"] < .008:
                continue
            start = max(0, min(duration - window, sample["time"] - window / 2))
            end = min(duration, start + window)
            if any(tiou((start, end), (x["start"], x["end"])) > .15 for x in selected):
                continue
            selected.append({"start": round(start, 6), "end": round(end, 6),
                             "peak_time": sample["time"], "proposal_score": sample["score"],
                             "reason": "visual_change" if config.strategy == "motion" else "uniform_coverage"})
            if len(selected) >= config.max_candidates:
                break
        return sorted(selected, key=lambda x: x["start"])
