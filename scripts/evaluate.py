"""Evaluate a fixed reference manifest; inference never receives its labels."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

from risksifter.candidates import MotionStrategy, scan
from risksifter.config import MiningConfig
from risksifter.evaluation import evaluate_events
from risksifter.pipeline import Pipeline
from risksifter.media import sha256

parser = argparse.ArgumentParser()
parser.add_argument("manifest", help="JSON: {videos:[{path,group_id,events:[{start,end}],label_source}]}")
parser.add_argument("--workspace", default="data/evaluation-workspace")
parser.add_argument("--output", default="data/results/evaluation.json")
parser.add_argument("--max-candidates", type=int, default=4)
parser.add_argument("--window-seconds", type=float, default=6)
args = parser.parse_args()
manifest_path = Path(args.manifest).resolve()
manifest = json.loads(manifest_path.read_text())
pipeline = Pipeline(args.workspace)
refs, predictions = [], {"motion": [], "uniform": []}
groups = {}
started = time.perf_counter()
for item in manifest["videos"]:
    group = item["group_id"]
    split = item.get("split", "test")
    if group in groups and groups[group] != split:
        raise ValueError("Trip/group leakage across splits")
    groups[group] = split
    if split != "test":
        continue
    path = (manifest_path.parent / item["path"]).resolve()
    if item.get("sha256") and sha256(path) != item["sha256"]:
        raise ValueError("Input hash mismatch")
    video = pipeline.import_video(path)
    refs.append({"video_id": video["id"], "duration": video["duration"], "events": item["events"],
                 "label_source": item["label_source"], "group_id": group})
    samples = scan(video["proxy_file"])
    for strategy in predictions:
        config = MiningConfig(strategy=strategy, max_candidates=args.max_candidates, window_seconds=args.window_seconds)
        candidates = MotionStrategy().propose(samples, video["duration"], config)
        predictions[strategy].extend({"video_id": video["id"], "start": p["start"], "end": p["end"], "score": p["proposal_score"]} for p in candidates)
result = {"manifest_sha256": sha256(manifest_path), "reference_metadata": refs,
          "configuration": {"budget": args.max_candidates, "window_seconds": args.window_seconds, "sample_fps": 2},
          "hardware": {"platform": platform.platform(), "python": platform.python_version()},
          "elapsed_seconds": time.perf_counter() - started,
          "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "results": {strategy: {str(th): evaluate_events(pred, refs, th) for th in [.1, .3, .5]} for strategy, pred in predictions.items()}}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: {t: {m: d[m] for m in ['precision','recall','f1','true_positives','false_positives','false_negatives']} for t,d in v.items()} for k,v in result['results'].items()}, indent=2))
