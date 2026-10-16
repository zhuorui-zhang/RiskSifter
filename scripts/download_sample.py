"""Download an official, MIT-licensed real driving sample; no API key required."""
import argparse
import json
import os
from pathlib import Path
from risksifter.media import ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/public")
args = parser.parse_args()
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download

revision = "4bff77c7254c654c28d4c2726186b4e825adccee"
filename = "compression_challenge/b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc"
file = hf_hub_download("commaai/comma2k19", filename, revision=revision, repo_type="dataset")
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
provenance = {"source": "https://huggingface.co/datasets/commaai/comma2k19", "revision": revision,
              "original_file": filename, "license": "MIT", "copyright": "Copyright (c) 2018 comma.ai",
              "citation": "Schafer et al., A Commute in Data: The comma2k19 Dataset, 2018, arXiv:1812.05752",
              "viewpoint": "ego-driving", "labels": "No accident ground truth supplied for this sample"}
mp4 = root / "comma2k19-driving.mp4"
# comma2k19 captures 20 fps. A raw HEVC stream has no container timestamps.
ffmpeg(["-y", "-r", "20", "-i", file, "-c:v", "copy", "-tag:v", "hvc1", "-movflags", "+faststart", mp4])
provenance["conversion"] = "Raw HEVC remuxed at comma2k19's documented 20 fps; no image content change"
(root / "comma2k19-driving.source.json").write_text(json.dumps(provenance, indent=2))
print(mp4.resolve())
