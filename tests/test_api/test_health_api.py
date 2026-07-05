import pytest
from fastapi.testclient import TestClient
from api.main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], list)
    assert "chest_segmentor" in data["models_loaded"]
    assert "detector" in data["models_loaded"]
