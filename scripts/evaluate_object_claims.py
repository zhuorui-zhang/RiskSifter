"""Compare model object mentions with publisher OpenLABEL, NOT accident detection.

Three fixed, raw, downsampled roadside clips. The input model never sees labels.
Object counts, localization and causal claims are deliberately not scored here.
"""
import argparse
import json
from pathlib import Path
import re

from risksifter.store import Store

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", default="data/raw-eval-workspace")
parser.add_argument("--subset", default="data/public/tum-raw/subset-manifest.json")
parser.add_argument("--output", default="data/results/object-claims.json")
args = parser.parse_args()
manifest_path = Path(args.subset)
manifest = json.loads(manifest_path.read_text())
store = Store(args.workspace)
classes = {"CAR": r"\b(cars?|automobiles?)\b", "TRUCK": r"\b(trucks?|lorr(?:y|ies))\b",
           "BUS": r"\bbus(?:es)?\b", "PEDESTRIAN": r"\b(pedestrians?|person|people)\b",
           "BICYCLE": r"\b(bicycles?|cyclists?)\b", "MOTORCYCLE": r"\bmotorcycles?\b"}
records = []
tp = fp = fn = 0
for clip in manifest["clips"]:
    video = next(v for v in store.all("videos") if v["sha256"] == clip["sha256"])
    event = next(e for e in store.all("events") if e["video_id"] == video["id"])
    truth = set()
    label_files = []
    viewed_indices = sorted({min(len(clip["frames"]) - 1, int(e["timestamp"] * 2 + 1e-6)) for e in event["evidence"]})
    for index in viewed_indices:
        frame = clip["frames"][index]
        if not frame["label_file"]:
            continue
        path = manifest_path.parent / frame["label_file"]
        label_files.append(frame["label_file"])
        data = json.loads(path.read_text())
        for value in data["openlabel"]["frames"].values():
            truth.update(o["object_data"]["type"] for o in value.get("objects", {}).values())
    truth &= classes.keys()
    claimed = {c for c, pattern in classes.items() if re.search(pattern, event["description"].lower())}
    if len(label_files) != len(viewed_indices):
        records.append({"sequence": clip["sequence"], "event_id": event["id"], "description": event["description"],
                        "scored": False, "reason": "Missing publisher labels: unknown, not a negative example",
                        "reference_classes": None, "claimed_classes": sorted(claimed)})
        continue
    tp += len(truth & claimed)
    fp += len(claimed - truth)
    fn += len(truth - claimed)
    records.append({"sequence": clip["sequence"], "event_id": event["id"], "description": event["description"], "scored": True,
                    "reference_classes": sorted(truth), "claimed_classes": sorted(claimed),
                    "unsupported_mentions": sorted(claimed - truth), "omitted_classes": sorted(truth - claimed),
                    "reference_files": label_files, "model": event["annotation"]["model"]})
result = {"task": "clip-level object class mention verification", "reference_source": "publisher OpenLABEL annotations",
          "num_clips": len(records), "num_scored_clips": sum(r["scored"] for r in records),
          "true_positive_mentions": tp, "false_positive_mentions": fp, "omitted_classes": fn,
          "micro_precision": tp / (tp + fp) if tp + fp else None, "micro_recall": tp / (tp + fn) if tp + fn else None,
          "records": records, "limitations": ["Only three 6s roadside clips sampled at 2fps; not a benchmark.",
              "Reference classes are the union of publisher labels for the 6 source frames selected as model evidence.",
              "Regex counts mentioned classes, including potential negated mentions; inspect records.",
              "Does not measure motion, accident detection, event recall, hallucinated relations or localization.",
              "No human verification of publisher annotation completeness; no train/test tuning performed."]}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"Scored {result['num_scored_clips']}/{len(records)} clips: {tp} supported mentions, {fp} unsupported, {fn} omitted. Saved to {args.output}")
