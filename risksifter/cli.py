import argparse
import json
from pathlib import Path

from .config import MiningConfig
from .pipeline import Pipeline
from .media import VIDEO_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description="RiskSifter — local video mining workbench")
    parser.add_argument("--workspace", default="workspace")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model-path", default="models/smolvlm2-2.2b")
    imp = sub.add_parser("import")
    imp.add_argument("path")
    mine = sub.add_parser("mine")
    mine.add_argument("video_id", help="video ID or 'all'")
    mine.add_argument("--annotator", choices=["none", "smolvlm"], default="none")
    mine.add_argument("--model-path", default="models/smolvlm2-2.2b")
    mine.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    mine.add_argument("--max-candidates", type=int, default=4)
    mine.add_argument("--window-seconds", type=float, default=6)
    mine.add_argument("--strategy", choices=["motion", "uniform"], default="motion")
    exp = sub.add_parser("export")
    exp.add_argument("--include-pending", action="store_true")
    sub.add_parser("doctor")
    sub.add_parser("list")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.workspace, args.model_path), host="127.0.0.1", port=args.port, access_log=False)
        return
    pipeline = Pipeline(args.workspace)
    if args.command == "import":
        path = Path(args.path).expanduser()
        paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS) if path.is_dir() else [path]
        for path in paths:
            video = pipeline.import_video(path)
            print(f"Imported {video['name']} ({video['id']}, {video['duration']:.1f}s)")
    elif args.command == "mine":
        from .annotators import make_annotator
        config = MiningConfig(annotator=args.annotator, model_path=args.model_path, device=args.device,
                              max_candidates=args.max_candidates, window_seconds=args.window_seconds, strategy=args.strategy)
        ids = [v["id"] for v in pipeline.store.all("videos") if v["status"] == "ready"] if args.video_id == "all" else [args.video_id]
        backend = make_annotator(config)
        for video_id in ids:
            result = pipeline.mine(video_id, config, lambda msg: print(msg, flush=True), backend)
            print(f"{video_id}: {result['status']}, {len(result['event_ids'])} candidates"
                  + (" (reused)" if result.get('cached') else f", {result['elapsed_seconds']:.1f}s"))
    elif args.command == "export":
        from .export import export_dataset
        print(json.dumps(export_dataset(pipeline.store, args.include_pending), ensure_ascii=False, indent=2))
    elif args.command == "list":
        for video in pipeline.store.all("videos"):
            print(f"{video['id']}  {video['status']}  {video['name']}")
    elif args.command == "doctor":
        import platform
        import psutil
        import imageio_ffmpeg
        report = {"python": platform.python_version(), "platform": platform.platform(),
                  "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
                  "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(), "workspace": str(pipeline.store.root)}
        try:
            import torch
            report.update(torch=torch.__version__, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available())
        except ImportError:
            report["torch"] = "not installed (lightweight mode available)"
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
