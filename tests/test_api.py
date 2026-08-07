from fastapi.testclient import TestClient

from visionguard.api import app


def test_health_and_cameras():
    with TestClient(app) as client:
        health = client.get("/health")
        cameras = client.get("/api/v1/cameras")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert cameras.status_code == 200
    assert any(camera["id"] == "demo_warehouse" for camera in cameras.json())
