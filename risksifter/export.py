from copy import deepcopy
import json
from pathlib import Path
import shutil
import uuid

from .media import clip, sha256
from .store import now


def export_dataset(store, include_pending=False, event_ids=None):
    allowed = {"approved", "pending"} if include_pending else {"approved"}
    all_events = store.all("events") if event_ids is not None else store.latest_events()
    if event_ids is not None:
        missing = set(event_ids) - {e["id"] for e in all_events}
        if missing:
            raise ValueError("Unknown selected event IDs")
    events = [e for e in all_events if e["review_status"] in allowed and (event_ids is None or e["id"] in event_ids)]
    if not events:
        raise ValueError("没有符合导出条件的事件。请确认事件，或显式包含待审核伪标签。")
    export_id = "v1-" + uuid.uuid4().hex[:12]
    exports = store.root / "exports"
    exports.mkdir(exist_ok=True)
    staging = exports / ("." + export_id)
    staging.mkdir()
    (staging / "clips").mkdir()
    records, sources, runs = [], {}, {}
    try:
        for original in sorted(events, key=lambda e: (e["video_id"], e["start"])):
            event = deepcopy(original)
            video = store.get("videos", event["video_id"])
            clip_path = Path("clips") / (event["id"] + ".mp4")
            clip(video["proxy_file"], staging / clip_path, event["start"], event["end"])
            event["clip"] = str(clip_path)
            event["source_sha256"] = video["sha256"]
            event["label_kind"] = "human_reviewed" if event["review_status"] == "approved" else "automatic_pseudo_label"
            for item in event["evidence"]:
                dest = Path("evidence") / event["id"] / Path(item["file"]).name
                (staging / dest).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item["file"], staging / dest)
                item["file"] = str(dest)
                item["inside_reviewed_interval"] = event["start"] <= item["timestamp"] < event["end"]
            records.append(event)
            sources[video["id"]] = {k: video[k] for k in ("id", "name", "sha256", "proxy_sha256", "provenance", "timebase", "duration")}
            run = store.get("runs", event["run_id"])
            run = deepcopy(run)
            run["config"]["model_path"] = Path(run["config"]["model_path"]).name
            run.pop("traceback", None)
            runs[run["id"]] = run
        (staging / "annotations.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in records))
        manifest = {"schema_version": "1.0", "id": export_id, "created_at": now(), "event_count": len(records),
                    "includes_pending": include_pending, "sources": list(sources.values()), "runs": list(runs.values()),
                    "time_unit": "seconds", "interval_convention": "[start, end)",
                    "notes": ["Clips use silent 15fps proxies; original source hashes retained.",
                              "Model evidence remains attached to the original proposal after human boundary changes.",
                              "Source media retain their own licenses. No automatic anonymization is applied."]}
        notices = []
        for source in sources.values():
            provenance = source["provenance"]
            notice = f"Source: {source['name']} ({source['id']})\n" + json.dumps(provenance, ensure_ascii=False, indent=2)
            if provenance.get("license") == "MIT":
                terms = (Path(__file__).parent / "licenses" / "MIT.txt").read_text()
                notice += "\n" + terms.format(copyright=provenance.get("copyright", "Copyright: see original source"))
            if provenance.get("license") == "CC BY-NC-SA 4.0":
                notice += "\nDerived video clips and evidence images retain CC BY-NC-SA 4.0.\nhttps://creativecommons.org/licenses/by-nc-sa/4.0/\nChanges: normalized resolution/frame rate, silent excerpts, sampled evidence frames. No endorsement implied.\n"
            notices.append(notice)
        (staging / "SOURCE_LICENSES.txt").write_text("\n\n".join(notices))
        (staging / "README.txt").write_text("RiskSifter dataset v1\nSee manifest.json and SOURCE_LICENSES.txt for source licenses and lineage.\nPending annotations are NOT ground truth. Original video is not included.\n")
        manifest["files"] = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        final = exports / export_id
        staging.rename(final)
        shutil.make_archive(str(final), "zip", final)
        return {"id": export_id, "directory": str(final), "zip": str(final) + ".zip", "event_count": len(records)}
    except Exception:
        # Partial exports are hidden and cannot be downloaded as completed datasets.
        shutil.rmtree(staging, ignore_errors=True)
        raise
