"""Temporal event evaluation with one-to-one matching; no self-generated rewards."""
from .candidates import tiou


def evaluate_events(predictions, references, threshold=.3):
    """Score-ranked one-to-one temporal matching; video IDs must be leakage-free groups."""
    if not 0 < threshold <= 1:
        raise ValueError("tIoU threshold must be in (0,1]")
    gt = {}
    total = 0
    for row in references:
        if row["video_id"] in gt:
            raise ValueError("Duplicate reference video_id")
        for interval in row["events"]:
            if not 0 <= interval["start"] < interval["end"] <= row["duration"]:
                raise ValueError("Invalid reference interval")
        gt[row["video_id"]] = row
        total += len(row["events"])
    matched = {video_id: set() for video_id in gt}
    matches, false_positives, overlap = [], [], []
    for pred in sorted(predictions, key=lambda p: -p.get("score", 0)):
        if pred["video_id"] not in gt:
            raise ValueError("Prediction video missing from reference set")
        if not 0 <= pred["start"] < pred["end"] <= gt[pred["video_id"]]["duration"]:
            raise ValueError("Invalid predicted interval")
        options = [(tiou((pred["start"], pred["end"]), (ref["start"], ref["end"])), i)
                   for i, ref in enumerate(gt[pred["video_id"]]["events"]) if i not in matched[pred["video_id"]]]
        best, index = max(options, default=(0, -1))
        if best >= threshold:
            matched[pred["video_id"]].add(index)
            matches.append({"prediction": pred, "reference_index": index, "tiou": best})
            overlap.append(best)
        else:
            false_positives.append(pred)
    tp, fp, fn = len(matches), len(false_positives), total - len(matches)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / total if total else None
    duration = sum(row["duration"] for row in references)
    return {"threshold": threshold, "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mean_matched_tiou": sum(overlap) / len(overlap) if overlap else None,
            "video_hours": duration / 3600, "matched_events_per_video_hour": tp * 3600 / duration if duration else None,
            "proposal_seconds": sum(p["end"] - p["start"] for p in predictions),
            "matches": matches, "unmatched_predictions": false_positives,
            "unmatched_references": [{"video_id": key, "event": ref} for key, row in gt.items()
                                      for i, ref in enumerate(row["events"]) if i not in matched[key]],
            "note": "Candidate localization metrics, not calibrated risk classification. Label quality is determined by the supplied references."}
