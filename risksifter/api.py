from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import MiningConfig
from .export import export_dataset
from .media import VIDEO_EXTENSIONS, clip
from .pipeline import Pipeline, search_events
from .store import RecordNotFound, now


class ImportRequest(BaseModel):
    path: str


class MineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str
    annotator: str = "none"
    max_candidates: int = Field(default=4, ge=1, le=50)
    window_seconds: float = Field(default=6, ge=2, le=30)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    revision: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    description: str = Field(min_length=1, max_length=8000)
    query: str = Field(min_length=1, max_length=8000)
    tags: list[str] = Field(default_factory=list, max_length=30)
    review_status: str
    review_note: str = Field(default="", max_length=8000)


class ExportRequest(BaseModel):
    include_pending: bool = False
    event_ids: list[str] | None = None


def create_app(root="workspace", model_path="models/smolvlm2-2.2b"):
    pipeline = Pipeline(root)
    store = pipeline.store
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="risksifter")
    clip_lock = threading.Lock()
    # No zombie jobs after restart; user can retry using preserved sources/evidence.
    for job in store.all("jobs"):
        if job["status"] in {"queued", "running"}:
            job.update(status="interrupted", message="服务已重启，请重新提交；已完成的索引与事件会复用")
            store.put("jobs", job)

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="RiskSifter", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        if host not in {"localhost", "127.0.0.1", "testserver"}:
            return JSONResponse({"detail": "Local access only"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("sec-fetch-site") == "cross-site" or (origin and origin != f"{request.url.scheme}://{request.headers.get('host')}"):
                return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(RecordNotFound)
    async def missing(request, exc):
        return JSONResponse({"detail": "Record not found"}, status_code=404)

    def submit(kind, action):
        job = {"id": uuid.uuid4().hex, "kind": kind, "status": "queued", "message": "排队中", "created_at": now()}
        store.put("jobs", job)

        def progress(message):
            job["message"] = message
            store.put("jobs", job)

        def worker():
            job.update(status="running", started_at=now())
            store.put("jobs", job)
            try:
                result = action(progress)
                job.update(status="completed", result=result, message="完成")
                if isinstance(result, dict) and result.get("status") == "partial":
                    job.update(status="partial", message="部分处理未完成；请查看错误详情并重试")
            except Exception as exc:
                job.update(status="failed", error=str(exc), message=str(exc))
            finally:
                job["finished_at"] = now()
                store.put("jobs", job)
        executor.submit(worker)
        return job

    @app.get("/api/status")
    def status():
        import psutil
        return {"version": "0.1.0", "workspace": str(store.root), "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
                "model_available": (Path(model_path) / "config.json").is_file() and any(Path(model_path).glob("*.safetensors")),
                "model_name": Path(model_path).name,
                "search_mode": "lexical_bilingual", "offline": True}

    @app.get("/api/videos")
    def videos():
        return store.all("videos")

    @app.post("/api/import")
    def import_local(body: ImportRequest):
        path = Path(body.path).expanduser().resolve()
        if not path.exists():
            raise ValueError("路径不存在")
        paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS) if path.is_dir() else [path]
        if not paths:
            raise ValueError("文件夹中没有支持的视频")
        if len(paths) > 500:
            raise ValueError("Phase 1 每批最多 500 个视频，请缩小文件夹范围")

        def action(progress):
            imported, errors = [], []
            progress(f"正在导入 {len(paths)} 个视频")
            for p in paths:
                try:
                    imported.append(pipeline.import_video(p)["id"])
                except (OSError, ValueError) as exc:
                    errors.append({"file": p.name, "error": str(exc)})
            return {"video_ids": imported, "errors": errors, "status": "partial" if errors else "completed"}
        return submit("import", action)

    @app.post("/api/upload")
    def upload(file: UploadFile):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError("不支持的文件类型")
        folder = store.root / "uploads"
        folder.mkdir(exist_ok=True)
        dest = folder / (uuid.uuid4().hex + suffix)
        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024**3:
                        raise ValueError("浏览器上传上限 2GB；更大文件请使用本地路径")
                    out.write(chunk)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        finally:
            file.file.close()

        def action(progress):
            try:
                progress("正在导入视频并生成浏览器播放代理")
                return pipeline.import_video(dest, display_name=Path(file.filename or "video").name)
            finally:
                dest.unlink(missing_ok=True)
        return submit("import", action)

    @app.post("/api/mine")
    def mine(body: MineRequest):
        config = MiningConfig(annotator=body.annotator, model_path=model_path,
                              max_candidates=body.max_candidates, window_seconds=body.window_seconds)
        store.get("videos", body.video_id)
        return submit("mine", lambda progress: pipeline.mine(body.video_id, config, progress))

    @app.get("/api/events")
    def events(q: str = "", video_id: str = "", review_status: str = "", latest_only: bool = True):
        result = store.latest_events() if latest_only else store.all("events")
        if video_id:
            result = [e for e in result if e["video_id"] == video_id]
        if review_status:
            result = [e for e in result if e["review_status"] == review_status]
        return search_events(result, q)

    @app.get("/api/events/{event_id}")
    def event(event_id: str):
        return store.get("events", event_id)

    @app.patch("/api/events/{event_id}")
    def review(event_id: str, body: ReviewRequest):
        changes = body.model_dump(exclude={"revision"})
        return store.review(event_id, changes, body.revision)

    @app.get("/api/jobs")
    def jobs():
        return store.all("jobs")[:30]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return store.get("jobs", job_id)

    @app.get("/api/videos/{video_id}/media")
    def video_media(video_id: str):
        video = store.get("videos", video_id)
        return FileResponse(video["proxy_file"], media_type="video/mp4")

    @app.get("/api/events/{event_id}/evidence/{index}")
    def evidence(event_id: str, index: int):
        event = store.get("events", event_id)
        if not 0 <= index < len(event["evidence"]):
            raise HTTPException(404, "Evidence not found")
        return FileResponse(event["evidence"][index]["file"], media_type="image/jpeg")

    @app.get("/api/events/{event_id}/clip")
    def event_clip(event_id: str):
        event = store.get("events", event_id)
        video = store.get("videos", event["video_id"])
        path = store.root / "clips" / f"{event_id}-r{event['revision']}.mp4"
        with clip_lock:
            if not path.is_file():
                clip(video["proxy_file"], path, event["start"], event["end"])
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.post("/api/exports")
    def export(body: ExportRequest):
        return submit("export", lambda progress: export_dataset(store, body.include_pending, body.event_ids))

    @app.get("/api/exports/{export_id}")
    def download_export(export_id: str):
        import re
        if not re.fullmatch(r"v1-[a-f0-9]{12}", export_id):
            raise HTTPException(404)
        path = store.root / "exports" / (export_id + ".zip")
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    return app
