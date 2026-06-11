import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], list)
