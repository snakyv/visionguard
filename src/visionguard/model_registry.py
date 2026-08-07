from __future__ import annotations

from pathlib import Path

from visionguard.config import ROOT


def list_models() -> list[dict]:
    model_dir = ROOT / "models"
    artifacts: list[dict] = []
    for path in sorted(model_dir.iterdir()):
        if path.name == "README.md":
            continue
        if path.is_file() and path.suffix.lower() in {".pt", ".onnx"}:
            artifacts.append({
                "name": path.name,
                "format": path.suffix.lower().lstrip("."),
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
            })
        elif path.is_dir() and path.name.endswith("_openvino_model"):
            size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
            artifacts.append({
                "name": path.name,
                "format": "openvino",
                "path": str(path.relative_to(ROOT)),
                "size_bytes": size,
            })
    return artifacts
