"""Explicit network step. All inference afterwards uses local_files_only=True."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--revision", default="482adb537c021c86670beed01cd58990d01e72e4")
parser.add_argument("--output", default="models/smolvlm2-2.2b")
args = parser.parse_args()
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download

revision = HfApi().model_info(args.repo, revision=args.revision).sha
snapshot_download(args.repo, revision=revision, local_dir=args.output,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"])
root = Path(args.output)
# Hub handles download integrity; provenance only needs repository and revision.
(root / "download_manifest.json").write_text(json.dumps({"repo_id": args.repo, "revision": revision}, indent=2))
print(root.resolve())
