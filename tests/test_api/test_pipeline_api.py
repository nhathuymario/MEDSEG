import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
from api.main import app

def test_pipeline_endpoint():
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/api/pipeline",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
    
    assert response.status_code == 503
    data = response.json()
    assert "ISIC segmentation checkpoint has not been trained/loaded" in data["detail"]
