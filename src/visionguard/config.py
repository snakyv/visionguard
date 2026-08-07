from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class CameraDefinition:
    id: str
    source: str | int
    backend: str = "synthetic"
    tracker: str = "centroid"
    enabled: bool = False
    model: str = "yolo26n.pt"
    confidence: float = 0.35
    device: str | None = None


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_cameras(path: str | Path | None = None) -> list[CameraDefinition]:
    config_path = path or os.getenv("VISIONGUARD_CONFIG", "configs/cameras.yaml")
    raw = load_yaml(config_path)
    result = []
    for item in raw.get("cameras", []):
        source = item.get("source", 0)
        if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")):
            source = str(resolve_path(source))
        result.append(
            CameraDefinition(
                id=str(item["id"]),
                source=source,
                backend=str(item.get("backend", "synthetic")),
                tracker=str(item.get("tracker", "centroid")),
                enabled=bool(item.get("enabled", False)),
                model=str(item.get("model", "yolo26n.pt")),
                confidence=float(item.get("confidence", 0.35)),
                device=item.get("device"),
            )
        )
    return result
