import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from risksifter.api import create_app
from risksifter.candidates import tiou, MotionStrategy
from risksifter.config import MiningConfig
from risksifter.export import export_dataset
from risksifter.media import probe, sha256
from risksifter.pipeline import Pipeline, search_events


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "motion.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 180))
    rng = np.random.default_rng(42)
    background = rng.integers(20, 100, (180, 320, 3), dtype=np.uint8)
    for i in range(180):
        frame = background.copy()
        if 45 <= i < 105:
            cv2.rectangle(frame, ((i - 45) * 4, 50), ((i - 45) * 4 + 50, 145), (230, 210, 200), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def mined(tmp_path, source):
    pipeline = Pipeline(tmp_path / "library")
    video = pipeline.import_video(source)
    config = MiningConfig(max_candidates=2)
    run = pipeline.mine(video["id"], config)
    return pipeline, video, config, run


def test_real_decode_mining_dedup_and_resume(mined, source):
    pipeline, video, config, run = mined
    assert 11.9 <= video["duration"] <= 12.1
    assert pipeline.import_video(source)["id"] == video["id"]
    assert len(pipeline.store.all("videos")) == 1
    assert run["status"] == "completed"
    assert 1 <= len(run["event_ids"]) <= 2
    assert pipeline.mine(video["id"], config)["cached"] is True
    for event in pipeline.store.all("events"):
        assert event["review_status"] == "pending"
        assert event["annotation"]["model"]["backend"] == "none"
        assert event["annotation"]["viewed_evidence"] == []
        assert event["risk_level"] is event["ttc_seconds"] is None
        for e in event["evidence"]:
            assert event["start"] <= e["timestamp"] < event["end"]
            assert sha256(e["file"]) == e["sha256"]
    assert any(e["start"] < 7 and e["end"] > 3 for e in pipeline.store.all("events"))


def test_review_history_conflict_and_boundary_validation(mined):
    pipeline, video, _, run = mined
    store = pipeline.store
    event = store.get("events", run["event_ids"][0])
    original_annotation = event["annotation"]
    with pytest.raises(ValueError, match="Boundaries"):
        store.review(event["id"], {"start": 5, "end": 4}, 0)
    edited = store.review(event["id"], {"review_status": "approved", "description": "Reviewed moving rectangle", "start": 0, "end": 2}, 0)
    assert edited["revision"] == 1 and edited["boundary_changed"]
    assert edited["label_kind"] == "human_reviewed"
    assert edited["annotation"] == original_annotation
    assert edited["history"][0]["before"]["review_status"] == "pending"
    with pytest.raises(ValueError, match="Revision conflict"):
        store.review(event["id"], {"review_status": "rejected"}, 0)
    assert store.save_generated_event(event) is False
    assert store.get("events", event["id"])["description"] == "Reviewed moving rectangle"


def test_export_playable_clips_hashes_and_review_policy(mined):
    pipeline, _, _, run = mined
    store = pipeline.store
    with pytest.raises(ValueError, match="没有"):
        export_dataset(store)
    selected = run["event_ids"][0]
    store.review(selected, {"review_status": "approved", "start": 1, "end": 3.5}, 0)
    exported = export_dataset(store)
    assert exported["event_count"] == 1
    root = Path(exported["directory"])
    event = json.loads((root / "annotations.jsonl").read_text().splitlines()[0])
    assert event["label_kind"] == "human_reviewed"
    assert abs(probe(root / event["clip"])["duration"] - 2.5) < .1
    assert event["source_sha256"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(sha256(root / name) == digest for name, digest in manifest["files"].items())
    assert str(store.root) not in (root / "annotations.jsonl").read_text()
    with zipfile.ZipFile(exported["zip"]) as archive:
        assert "manifest.json" in archive.namelist()
    store.review(selected, {"review_status": "rejected"}, 1)
    pending = [e for e in store.all("events") if e["review_status"] == "pending"]
    if pending:
        export2 = export_dataset(store, include_pending=True)
        assert export2["event_count"] == len(pending)


def test_missing_model_is_explicit_partial_and_retryable(mined):
    pipeline, video, _, _ = mined
    config = MiningConfig(annotator="smolvlm", model_path="/missing-model", max_candidates=1)
    run = pipeline.mine(video["id"], config)
    assert run["status"] == "partial" and run["warnings"]
    event = pipeline.store.get("events", run["event_ids"][0])
    assert event["annotation_error"] and event["annotation"]["model"]["backend"] == "none"
    assert "cached" not in pipeline.mine(video["id"], config)


def test_annotation_bug_is_reported_as_failed_not_downgraded(mined):
    pipeline, video, _, _ = mined

    class BrokenAnnotator:
        def annotate(self, evidence, proposal):
            raise TypeError("Unexpected annotation bug")

    with pytest.raises(TypeError, match="Unexpected annotation bug"):
        pipeline.mine(video["id"], MiningConfig(max_candidates=1), annotator=BrokenAnnotator())
    failed = next(r for r in pipeline.store.all("runs") if r["status"] == "failed")
    assert failed["error"] == "Unexpected annotation bug"
    assert not any(e["run_id"] == failed["id"] for e in pipeline.store.all("events"))


def test_static_video_produces_no_motion_risk(tmp_path):
    path = tmp_path / "static.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (160, 90))
    for _ in range(60):
        writer.write(np.full((90, 160, 3), 64, dtype=np.uint8))
    writer.release()
    p = Pipeline(tmp_path / "static-library")
    v = p.import_video(path)
    assert p.mine(v["id"])["event_ids"] == []


def test_search_is_transparent_and_nonmatching_queries_are_empty():
    events = [{"id": "1", "description": "A pedestrian crosses the road.", "query": "Find a pedestrian", "tags": ["pedestrian"]},
              {"id": "2", "description": "Cars moving through an intersection.", "query": "Find cars", "tags": ["vehicle"]}]
    assert search_events(events, "查找行人")[0]["id"] == "1"
    assert search_events(events, "unicorn") == []
    assert len(search_events(events, "")) == 2


def test_api_media_review_errors_and_cross_origin(mined):
    pipeline, video, _, run = mined
    with TestClient(create_app(pipeline.store.root)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["offline"] is True
        assert client.get("/api/events?video_id=" + video["id"]).json()
        response = client.get(f"/api/videos/{video['id']}/media", headers={"Range": "bytes=0-99"})
        assert response.status_code == 206 and len(response.content) == 100
        event = pipeline.store.get("events", run["event_ids"][0])
        body = {k: event[k] for k in ["revision", "start", "end", "description", "query", "tags", "review_status", "review_note"]}
        body["review_status"] = "approved"
        assert client.patch("/api/events/" + event["id"], json=body).status_code == 200
        assert client.patch("/api/events/" + event["id"], json=body).status_code == 400
        assert client.get(f"/api/events/{event['id']}/evidence/-1").status_code == 404
        assert client.get("/api/events/nonexistent").status_code == 404
        assert client.post("/api/import", json={"path": "/etc/passwd"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
        assert client.get("/api/exports/no-such-export").status_code == 404
        bad = {**body, "revision": 1, "start": -5}
        assert client.patch("/api/events/" + event["id"], json=bad).status_code == 422


def test_import_failure_retained_and_jobs_survive_restart(tmp_path):
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"not a video")
    p = Pipeline(tmp_path / "w")
    with pytest.raises(ValueError):
        p.import_video(corrupt)
    assert p.store.all("videos")[0]["status"] == "failed"
    p.store.put("jobs", {"id": "dead-job", "status": "running"})
    with TestClient(create_app(p.store.root)) as client:
        assert client.get("/api/jobs/dead-job").json()["status"] == "interrupted"


def test_temporal_overlap_and_budget():
    assert tiou((0, 6), (3, 9)) == pytest.approx(1 / 3)
    assert tiou((0, 1), (2, 3)) == 0
    proposals = MotionStrategy().propose([{"time": i, "score": .1} for i in range(50)], 50, MiningConfig(max_candidates=3))
    assert len(proposals) == 3
    assert all(0 <= p["start"] < p["end"] <= 50 for p in proposals)


def test_new_run_keeps_history_without_inflating_latest_count(mined):
    pipeline, video, _, previous = mined
    new = pipeline.mine(video["id"], MiningConfig(strategy="uniform", max_candidates=1))
    assert {e["run_id"] for e in pipeline.store.latest_events()} == {new["id"]}
    assert pipeline.store.get("events", previous["event_ids"][0])
    with TestClient(create_app(pipeline.store.root)) as client:
        assert len(client.get("/api/events").json()) == 1
        assert len(client.get("/api/events?latest_only=false").json()) > 1
