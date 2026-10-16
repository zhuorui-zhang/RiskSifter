"""Run a frozen evidence set through alternative models; never edits user reviews."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

from risksifter.annotators import LocalVLMAnnotator
from risksifter.config import MiningConfig
from risksifter.store import Store

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", default="workspace")
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="auto")
parser.add_argument("--limit", type=int, default=4)
parser.add_argument("--evidence-manifest", help="Prior comparison JSON, to reuse exactly the same event IDs")
args = parser.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
store = Store(args.workspace)
backend = LocalVLMAnnotator(MiningConfig(annotator="smolvlm", model_path=args.model_path, device=args.device))
# Stable selection, two from each video. Independent of generated text.
grouped = {}
for event in store.all("events"):
    grouped.setdefault(event["video_id"], []).append(event)
selected = []
for events in grouped.values():
    distinct = {}
    for event in sorted(events, key=lambda e: (e["start"], e["created_at"])):
        distinct.setdefault((event["start"], event["end"]), event)
    selected.extend(list(distinct.values())[:2])
if args.evidence_manifest:
    prior = json.loads(Path(args.evidence_manifest).read_text())
    selected = [store.get("events", e["event_id"]) for e in prior["events"]]
selected = selected[:args.limit]
started = time.perf_counter()
result = {"model_path": args.model_path, "platform": platform.platform(), "events": []}
for event in selected:
    annotation = backend.annotate(event["evidence"], event["proposal"])
    result["events"].append({"event_id": event["id"], "video_id": event["video_id"], "start": event["start"],
                             "end": event["end"], "evidence": event["evidence"], "annotation": annotation})
    result["elapsed_seconds"] = time.perf_counter() - started
    result["max_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"Compared {len(selected)} candidates; saved results to {args.output}")
