import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(name="client")
def _client():
    return TestClient(app)


@pytest.fixture(name="sample_image_bytes")
def _sample_image_bytes():
    img = Image.new("RGB", (100, 100), color="green")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data


class TestHomeEndpoint:

    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Pet Breed Classification API" in response.text


class TestClassesEndpoint:

    @patch("logic.image_processor.load_model")
    def test_get_classes(self, mock_load_model, client):
        mock_session = MagicMock()
        class_labels = ["Abyssinian", "Bengal", "Persian"]
        mock_load_model.return_value = (mock_session, class_labels, "input")

        response = client.get("/api/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "count" in data
        assert isinstance(data["classes"], list)
        assert data["count"] > 0


class TestPredictEndpoint:

    @patch("logic.image_processor.load_model")
    def test_predict_with_valid_image(self, mock_load_model, client, sample_image_bytes):
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.2, 0.7]])]
        class_labels = ["Abyssinian", "Bengal", "Persian"]
        mock_load_model.return_value = (mock_session, class_labels, "input")

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "predicted_breed" in data
        assert data["filename"] == "test.jpg"

    def test_predict_without_file(self, client):
        response = client.post("/api/predict")
        assert response.status_code == 422

    def test_predict_with_non_image_file(self, client):
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 400


class TestResizeEndpoint:

    def test_resize_with_default_dimensions(self, client, sample_image_bytes):
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/resize", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "new_size" in data

    def test_resize_with_custom_dimensions(self, client, sample_image_bytes):
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/resize?width=50&height=50", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["new_size"]["width"] == 50
        assert data["new_size"]["height"] == 50

    def test_resize_with_invalid_dimensions(self, client, sample_image_bytes):
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/resize?width=-1&height=50", files=files)
        assert response.status_code == 400


class TestPreprocessEndpoint:

    def test_preprocess_with_default_options(self, client, sample_image_bytes):
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/preprocess", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "preprocessing" in data

    def test_preprocess_with_grayscale(self, client, sample_image_bytes):
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/preprocess?grayscale=true", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["preprocessing"]["grayscale"] is True

    def test_preprocess_without_normalize(self, client, sample_image_bytes):
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/preprocess?normalize=false", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["preprocessing"]["normalized"] is False
