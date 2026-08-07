from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from visionguard import __version__
from visionguard.config import ROOT, load_cameras
from visionguard.model_registry import list_models
from visionguard.runtime import Runtime


runtime = Runtime(load_cameras())


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start_enabled()
    yield
    runtime.stop_all()


app = FastAPI(
    title="VisionGuard API",
    version=__version__,
    description="Real-time multi-camera video analytics API",
    lifespan=lifespan,
)


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "cameras": len(runtime.workers)}


@app.get("/api/v1/cameras")
def cameras():
    return [
        {
            "id": worker.camera.id,
            "source": str(worker.camera.source),
            "backend": worker.camera.backend,
            "tracker": worker.camera.tracker,
            "model": worker.camera.model if worker.camera.backend == "yolo" else None,
            "running": worker.running,
            "last_error": worker.last_error,
            "metrics": {
                "frames": worker.metrics.frames,
                "fps": round(worker.metrics.fps, 2),
                "inference_ms": round(worker.metrics.inference_ms, 2),
                "events": worker.metrics.events,
                "objects": dict(worker.pipeline.analytics.class_counts) if worker.pipeline else {},
            },
        }
        for worker in runtime.workers.values()
    ]


@app.post("/api/v1/cameras/{camera_id}/start")
def start_camera(camera_id: str):
    worker = runtime.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    try:
        worker.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"id": camera_id, "running": worker.running}


@app.post("/api/v1/cameras/{camera_id}/stop")
def stop_camera(camera_id: str):
    worker = runtime.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    worker.stop()
    return {"id": camera_id, "running": False}


@app.get("/api/v1/events")
def events(limit: int = 100):
    limit = max(1, min(limit, 500))
    _, snapshot = runtime.event_snapshot()
    return [event.to_dict() for event in snapshot[:limit]]


@app.get("/api/v1/models")
def models():
    return list_models()


@app.get("/api/v1/metrics")
def metrics():
    return {
        camera_id: {
            "running": worker.running,
            "frames": worker.metrics.frames,
            "fps": round(worker.metrics.fps, 2),
            "inference_ms": round(worker.metrics.inference_ms, 2),
            "events": worker.metrics.events,
            "objects": dict(worker.pipeline.analytics.class_counts) if worker.pipeline else {},
            "last_error": worker.last_error,
        }
        for camera_id, worker in runtime.workers.items()
    }


@app.get("/api/v1/cameras/{camera_id}/mjpeg")
def mjpeg(camera_id: str):
    worker = runtime.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    async def stream():
        while True:
            frame = worker.latest_jpeg()
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    sequence = -1
    try:
        while True:
            current, snapshot = runtime.event_snapshot()
            if current != sequence:
                sequence = current
                await websocket.send_json({"sequence": current, "events": [event.to_dict() for event in snapshot[:50]]})
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
