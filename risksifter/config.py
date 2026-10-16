from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class MiningConfig:
    sample_fps: float = 2.0
    window_seconds: float = 6.0
    max_candidates: int = 4
    evidence_frames: int = 6
    strategy: str = "motion"
    annotator: str = "none"
    model_path: str = "models/smolvlm2-2.2b"
    device: str = "auto"

    def __post_init__(self):
        if not 0.5 <= self.sample_fps <= 5:
            raise ValueError("sample_fps must be in [0.5, 5]")
        if not 2 <= self.window_seconds <= 30:
            raise ValueError("window_seconds must be in [2, 30]")
        if not 1 <= self.max_candidates <= 50 or not 2 <= self.evidence_frames <= 16:
            raise ValueError("candidate/frame budget outside allowed range")
        if self.strategy not in {"motion", "uniform"}:
            raise ValueError("Unknown strategy")
        if self.annotator not in {"none", "smolvlm"}:
            raise ValueError("Unknown annotator")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unknown device")

    def signature(self):
        # Include code/prompt versions so repeat runs never silently reuse a new policy.
        from .annotators import PROMPT_VERSION
        body = {**asdict(self), "pipeline": "phase1-v1", "prompt": PROMPT_VERSION}
        if self.annotator != "none":
            from pathlib import Path
            model = Path(self.model_path)
            manifest = model / "download_manifest.json"
            if manifest.exists():
                body["model_manifest"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            # Also invalidate when local weights/config change without updating the manifest.
            body["model_files"] = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                                   for p in sorted(model.glob("*")) if p.is_file() and p.suffix in {".json", ".safetensors"}]
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
