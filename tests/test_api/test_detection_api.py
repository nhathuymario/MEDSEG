import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
from api.main import app

client = TestClient(app)


def test_detection_endpoint():
    # Create a dummy image
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    response = client.post(
        "/api/detect",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "boxes" in data
    assert "scores" in data
    assert "labels" in data
    assert "inference_time_ms" in data
