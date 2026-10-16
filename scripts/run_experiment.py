import argparse
import json
from pathlib import Path
import platform
import resource
import time

from risksifter.annotators import make_annotator
from risksifter.config import MiningConfig
from risksifter.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", default="workspace")
parser.add_argument("--model-path", default="models/smolvlm2-2.2b")
parser.add_argument("--device", default="auto")
parser.add_argument("--output", default="data/results/run.json")
parser.add_argument("--max-candidates", type=int, default=4)
parser.add_argument("videos", nargs="+")
args = parser.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
pipeline = Pipeline(args.workspace)
config = MiningConfig(annotator="smolvlm", model_path=args.model_path, device=args.device, max_candidates=args.max_candidates)
backend = make_annotator(config)
report = {"platform": platform.platform(), "python": platform.python_version(), "videos": [], "runs": []}
started = time.perf_counter()
for path in args.videos:
    t = time.perf_counter()
    v = pipeline.import_video(path)
    imported = time.perf_counter()-t
    run = pipeline.mine(v["id"], config, lambda message: print(message, flush=True), backend)
    report["videos"].append({"id": v["id"], "name": v["name"], "source_sha256": v["sha256"], "duration": v["duration"], "import_seconds_cached_or_new": imported})
    report["runs"].append(run)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["max_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f"Saved {len(report['runs'])} runs to {args.output}")
