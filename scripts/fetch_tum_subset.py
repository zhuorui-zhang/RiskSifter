"""Fetch a small fixed raw-frame/OpenLABEL subset from the official public archive.

No rendered boxes are present in input images. Original timestamps and CRCs are
preserved. Reconstructed clips use 2fps sample-and-hold, not original 25fps video.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib

import requests
from PIL import Image
from risksifter.media import ffmpeg, sha256


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP ranges so zipfile can read a tiny part of the 9GB archive."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position + offset if whence == 1 else self.size + offset)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size - 1 if size < 0 else min(self.size, self.position + size) - 1
        if end < self.position:
            return b""
        response = requests.get(self.url, auth=("Zd4rwsMwwXMaXiK", ""),
                                headers={"Range": f"bytes={self.position}-{end}"}, timeout=90)
        response.raise_for_status()
        if response.headers.get("Content-Range") != f"bytes {self.position}-{end}/{self.size}":
            raise ValueError("Expected exact range response; refusing a full archive download")
        self.position = end + 1
        return response.content

URL = "https://nx21496.your-storageshare.de/public.php/webdav/accid3nd_dataset/train1.zip"
SIZE = 9436619494
root = Path("data/public/tum-raw")
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(RemoteZipFile(URL, SIZE)) as archive:
    info = {i.filename: i for i in archive.infolist()}
sources = {name: (URL, SIZE) for name in info}


def stamp(name):
    parts = Path(name).stem.split("_")
    return int(parts[0]) + int(parts[1]) / 1e9


def fetch(item):
    destination = root / item.filename
    if destination.exists() and destination.stat().st_size == item.file_size:
        return str(destination)
    reader = RemoteZipFile(*sources[item.filename])
    reader.seek(item.header_offset)
    header = reader.read(30)
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    reader.seek(item.header_offset + 30 + name_len + extra_len)
    data = reader.read(item.compress_size)
    if item.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif item.compress_type != zipfile.ZIP_STORED:
        raise ValueError("Unsupported zip compression")
    if zlib.crc32(data) != item.CRC or len(data) != item.file_size:
        raise ValueError("ZIP CRC mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return str(destination)


manifest = {"source": URL, "license": "CC BY-NC-SA 4.0", "selection": "r09_s02/s03/s04, north_50mm, offsets 57..62.5 seconds at 2fps; fixed before model inference", "clips": []}
for sequence in ["r09_s02", "r09_s03", "r09_s04"]:
    folder = sequence + "/images/s040_camera_basler_north_50mm/"
    names = sorted(n for n in info if n.startswith(folder) and n.endswith(".jpg"))
    origin = min(stamp(n) for n in names)
    selected = [min(names, key=lambda n: abs(stamp(n) - (origin + 57 + i * .5))) for i in range(12)]
    labels = [n.replace("/images/", "/labels/").replace(".jpg", ".json") for n in selected]
    if any(n not in info for n in labels):
        next_url = URL.replace("train1.zip", "train2.zip")
        with zipfile.ZipFile(RemoteZipFile(next_url, 9436522698)) as archive:
            for item in archive.infolist():
                if item.filename not in info:
                    info[item.filename] = item
                    sources[item.filename] = (next_url, 9436522698)
    available = [info[n] for n in selected + labels if n in info]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, available))
    out = root / (sequence + ".mp4")
    frames = root / sequence / "sampled"
    frames.mkdir(exist_ok=True)
    for i, name in enumerate(selected):
        with Image.open(root / name) as im:
            im.save(frames / f"{i:03}.jpg")
    ffmpeg(["-y", "-framerate", "2", "-i", frames / "%03d.jpg", "-vf", "scale=960:-2", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out])
    clip = {"path": str(out.resolve()), "sequence": sequence, "sha256": sha256(out),
            "frames": [{"clip_timestamp": i * .5, "source_timestamp": stamp(n), "source_file": n,
                        "source_sha256": sha256(root / n), "label_file": labels[i] if labels[i] in info else None,
                        "label_archive": sources[labels[i]][0] if labels[i] in info else None} for i,n in enumerate(selected)]}
    manifest["clips"].append(clip)
    out.with_suffix(".source.json").write_text(json.dumps({"source": "https://accident-dataset.github.io/", "original_archive": URL,
        "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "Zimmer et al., Towards Vision Zero: The TUM Traffic Accid3nD Dataset, 2025",
        "sequence": sequence, "viewpoint": "roadside", "conversion": "12 raw frames nearest 57..62.5s at 2fps; sample-and-hold video, no interpolation",
        "source_frame_manifest": clip["frames"]}, indent=2))
    print(sequence, out, flush=True)
(root / "subset-manifest.json").write_text(json.dumps(manifest, indent=2))
