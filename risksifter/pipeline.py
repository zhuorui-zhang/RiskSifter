from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

from . import __version__
from .annotators import NoModelAnnotator, make_annotator
from .candidates import MotionStrategy, scan, STRATEGY_VERSION
from .config import MiningConfig
from .media import VIDEO_EXTENSIONS, sha256, normalize, extract_evidence
from .store import RecordNotFound, Store, now


class Pipeline:
    def __init__(self, root):
        self.store = Store(root)

    def import_video(self, path, provenance=None, display_name=None):
        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Not a supported local video: {source}")
        sidecar = source.with_suffix(".source.json")
        if provenance is None and sidecar.is_file():
            provenance = json.loads(sidecar.read_text())
        digest = sha256(source)
        video_id = digest[:24]
        folder = self.store.root / "videos" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        try:
            existing = self.store.get("videos", video_id)
            if existing["status"] == "ready" and (folder / "proxy.mp4").is_file():
                return existing
        except RecordNotFound:
            pass
        snapshot = folder / ("source" + source.suffix.lower())
        video = {"id": video_id, "sha256": digest, "name": display_name or source.name, "created_at": now(),
                 "source_file": str(snapshot), "proxy_file": str(folder / "proxy.mp4"),
                 "provenance": provenance or {"source": "local_user_import", "license": "user_supplied_unspecified"},
                 "status": "processing", "error": None}
        self.store.put("videos", video)
        try:
            if source != snapshot:
                shutil.copyfile(source, snapshot)
            if sha256(snapshot) != digest:
                raise ValueError("Source changed during import; retry with a stable file")
            metadata = normalize(snapshot, folder / "proxy.mp4")
            video.update(metadata)
            video["status"] = "ready"
            video["proxy_sha256"] = sha256(folder / "proxy.mp4")
            video["timebase"] = "seconds from start of silent 15fps review proxy; tolerance 1/15s"
        except Exception as exc:
            video.update(status="failed", error=str(exc))
            raise
        finally:
            self.store.put("videos", video)
        return video

    def mine(self, video_id, config=None, progress=None, annotator=None):
        config = config or MiningConfig()
        report = progress or (lambda message: None)
        video = self.store.get("videos", video_id)
        if video["status"] != "ready":
            raise ValueError("Video import must complete first")
        run_id = hashlib.sha256(f"{video_id}:{config.signature()}".encode()).hexdigest()[:24]
        try:
            old = self.store.get("runs", run_id)
            if old["status"] == "completed":
                return {**old, "cached": True}
        except RecordNotFound:
            pass
        started = time.perf_counter()
        run = {"id": run_id, "video_id": video_id, "config": asdict(config), "created_at": now(),
               "status": "running", "event_ids": [], "warnings": [], "strategy_version": STRATEGY_VERSION,
               "software_version": __version__}
        self.store.put("runs", run)
        try:
            report("正在建立低频运动索引")
            index_path = self.store.root / "videos" / video_id / f"motion-{config.sample_fps:g}-v1.json"
            if index_path.exists():
                samples = json.loads(index_path.read_text())
            else:
                samples = scan(video["proxy_file"], config.sample_fps)
                temp = index_path.with_suffix(".tmp")
                temp.write_text(json.dumps(samples))
                temp.replace(index_path)
            proposals = MotionStrategy().propose(samples, video["duration"], config)
            run["sample_count"] = len(samples)
            run["index_seconds"] = time.perf_counter() - started
            backend = annotator or make_annotator(config)
            report(f"分析 {len(proposals)} 个候选片段")
            for i, proposal in enumerate(proposals):
                event_id = hashlib.sha256(f"{run_id}:{i}:{proposal}".encode()).hexdigest()[:24]
                run["event_ids"].append(event_id)
                try:
                    old_event = self.store.get("events", event_id)
                    # Successful annotations and all human edits survive retries.
                    if old_event["revision"] > 0 or not old_event.get("annotation_error"):
                        continue
                except RecordNotFound:
                    pass
                evidence = extract_evidence(video["proxy_file"], self.store.root / "evidence" / event_id,
                                            proposal["start"], proposal["end"], config.evidence_frames)
                error = None
                try:
                    annotation = backend.annotate(evidence, proposal)
                except (OSError, ImportError) as exc:
                    # Missing/unreadable model assets can degrade explicitly;
                    # programming and inference errors must fail the run.
                    error = f"{type(exc).__name__}: {exc}"
                    annotation = NoModelAnnotator().annotate(evidence, proposal)
                    annotation["uncertainties"].append("VLM 失败，已保留可审核候选与真实证据")
                    if error not in run["warnings"]:
                        run["warnings"].append(error)
                        report(error)
                event = {"schema_version": "1.0", "id": event_id, "video_id": video_id, "run_id": run_id,
                         "created_at": now(), "updated_at": now(), "start": proposal["start"], "end": proposal["end"],
                         "proposal": proposal, "annotation": annotation, "annotation_error": error,
                         "description": annotation["description"], "query": annotation["query"], "tags": annotation["tags"],
                         "evidence": evidence, "review_status": "pending", "review_note": "", "revision": 0,
                         "boundary_changed": False, "history": [], "label_kind": "automatic_pseudo_label",
                         "risk_level": None, "ttc_seconds": None, "distance_meters": None}
                self.store.save_generated_event(event)
            run["status"] = "partial" if run["warnings"] else "completed"
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = str(exc)
            raise
        finally:
            run["elapsed_seconds"] = time.perf_counter() - started
            run["video_seconds"] = video["duration"]
            run["finished_at"] = now()
            self.store.put("runs", run)
        return run


SYNONYMS = {
    "行人": "pedestrian person people walking", "人行道": "sidewalk", "骑行": "cyclist bicycle bike",
    "自行车": "cyclist bicycle bike", "摩托": "motorcycle", "车辆": "car vehicle truck bus",
    "汽车": "car vehicle", "卡车": "truck", "公交": "bus", "切入": "cut merge lane",
    "加塞": "cut merge lane", "碰撞": "collision crash collide", "事故": "collision crash accident",
    "路口": "intersection junction", "夜间": "night dark", "变道": "lane change merge",
}


def search_events(events, query):
    """Transparent lexical retrieval over saved annotations, with bilingual concept expansion."""
    import re
    if not query.strip():
        return events
    expanded = query.lower()
    for word, translations in SYNONYMS.items():
        if word in expanded:
            expanded += " " + translations
    stop = {"find", "all", "the", "a", "an", "of", "with", "showing", "clips", "video", "where", "and", "is"}
    tokens = [t for t in re.findall(r"[a-z]+|[\u4e00-\u9fff]+", expanded) if t not in stop]
    scored = []
    for event in events:
        text = " ".join([event["description"], event["query"], *event["tags"]]).lower()
        words = set(re.findall(r"[a-z]+", text))
        score = sum(t in words if t.isascii() else t in text for t in set(tokens))
        if score:
            scored.append({**event, "search_score": score})
    return sorted(scored, key=lambda e: -e["search_score"])
