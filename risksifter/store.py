from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def now():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class RecordNotFound(KeyError):
    """A missing stored record, distinct from a programming error in a dict lookup."""


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "library.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, video_id TEXT, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, video_id TEXT, run_id TEXT, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_video ON events(video_id);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, table, data):
        if table not in {"videos", "runs", "events", "jobs"}:
            raise ValueError("Unknown table")
        columns = ["id", "data"]
        values = [data["id"], dumps(data)]
        for extra in ("video_id", "run_id"):
            if extra in data and (table == "events" or (table == "runs" and extra == "video_id")):
                columns.append(extra)
                values.append(data[extra])
        with self.connect() as db:
            db.execute(f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values)

    def get(self, table, key):
        if table not in {"videos", "runs", "events", "jobs"}:
            raise ValueError("Unknown table")
        with self.connect() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE id=?", (key,)).fetchone()
        if row is None:
            raise RecordNotFound(f"{table}: {key}")
        return json.loads(row[0])

    def save_generated_event(self, event):
        # A reviewer may save while a retry is doing slow model inference.
        # Check again inside the write transaction, not only before inference.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM events WHERE id=?", (event["id"],)).fetchone()
            if row and json.loads(row[0])["revision"] > 0:
                return False
            db.execute("INSERT OR REPLACE INTO events(id,video_id,run_id,data) VALUES (?,?,?,?)",
                       (event["id"], event["video_id"], event["run_id"], dumps(event)))
        return True

    def all(self, table):
        if table not in {"videos", "runs", "events", "jobs"}:
            raise ValueError("Unknown table")
        with self.connect() as db:
            rows = db.execute(f"SELECT data FROM {table} ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def review(self, event_id, changes, expected_revision):
        allowed = {"start", "end", "description", "query", "tags", "review_status", "review_note"}
        if not set(changes).issubset(allowed):
            raise ValueError("Unknown review field")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM events WHERE id=?", (event_id,)).fetchone()
            if not row:
                raise RecordNotFound(event_id)
            event = json.loads(row[0])
            if event["revision"] != expected_revision:
                raise ValueError("Revision conflict: reload the event before saving")
            edited = {**event, **changes}
            video = json.loads(db.execute("SELECT data FROM videos WHERE id=?", (event["video_id"],)).fetchone()[0])
            if not 0 <= edited["start"] < edited["end"] <= video["duration"]:
                raise ValueError("Boundaries must satisfy 0 <= start < end <= video duration")
            if edited["review_status"] not in {"pending", "approved", "rejected"}:
                raise ValueError("Unknown review status")
            if not edited["description"].strip() or not edited["query"].strip():
                raise ValueError("Description and query cannot be empty")
            edited["revision"] += 1
            edited["label_kind"] = "human_reviewed" if edited["review_status"] == "approved" else "automatic_pseudo_label"
            edited["updated_at"] = now()
            edited["history"] = event["history"] + [{"time": now(), "actor": "local_reviewer", "revision": edited["revision"],
                                                      "before": {k: event.get(k) for k in changes}, "after": changes}]
            # Preserve original model evidence and output, even after boundary edits.
            edited["boundary_changed"] = (edited["start"], edited["end"]) != (event["proposal"]["start"], event["proposal"]["end"])
            db.execute("UPDATE events SET data=? WHERE id=?", (dumps(edited), event_id))
        return edited

    def latest_events(self):
        """Keep historical runs addressable, but don't count repeated runs as new events."""
        latest = {}
        for run in sorted(self.all("runs"), key=lambda r: r["created_at"], reverse=True):
            if run["status"] in {"completed", "partial"}:
                latest.setdefault(run["video_id"], run["id"])
        return [e for e in self.all("events") if latest.get(e["video_id"]) == e["run_id"]]
