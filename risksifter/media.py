import hashlib
import math
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffmpeg(args):
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *map(str, args)],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode:
        raise ValueError(f"Video conversion failed: {result.stderr[-2000:]}")


def normalize(source, destination):
    """A silent CFR review proxy gives all downstream stages one timestamp clock."""
    destination = Path(destination)
    temp = destination.with_suffix(".tmp.mp4")
    ffmpeg(["-y", "-i", source, "-map", "0:v:0", "-an", "-vf",
            "fps=15,scale=w='min(960,iw)':h=-2", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", temp])
    metadata = probe(temp)
    temp.replace(destination)
    return metadata


def probe(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("No decodable video frames")
        return {"fps": fps, "frames": count, "duration": count / fps,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def frame_at(path, seconds):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = max(0, min(int(round(seconds * fps)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not decode evidence at {seconds:.3f}s")
        return frame, n / fps
    finally:
        cap.release()


def extract_evidence(proxy, folder, start, end, count):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    evidence = []
    for i, seconds in enumerate(np.linspace(start, max(start, end - 1 / 15), count)):
        frame, actual = frame_at(proxy, float(seconds))
        file = folder / f"{i:02}.jpg"
        if not cv2.imwrite(str(file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise ValueError("Unable to save evidence")
        evidence.append({"timestamp": round(actual, 6), "file": str(file), "sha256": sha256(file)})
    return evidence


def clip(proxy, destination, start, end):
    if not 0 <= start < end:
        raise ValueError("Invalid clip boundaries")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.mp4")
    ffmpeg(["-y", "-ss", start, "-i", proxy, "-t", end - start, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp])
    probe(temp)
    temp.replace(destination)
