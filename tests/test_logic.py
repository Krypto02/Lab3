import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from logic.image_processor import (
    predict_class,
    preprocess_image,
    resize_image,
)


@pytest.fixture
def sample_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.new("RGB", (224, 224), color="red")
        img.save(tmp.name)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def output_path():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestPredictClass:

    @patch("logic.image_processor.load_model")
    def test_predict_class_returns_string(self, mock_load_model, sample_image):
        # Mock ONNX model
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.2, 0.7]])]
        mock_load_model.return_value = (mock_session, ["Class1", "Class2", "Class3"], "input")

        breed, confidence = predict_class(sample_image)
        assert isinstance(breed, str)
        assert len(breed) > 0
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @patch("logic.image_processor.load_model")
    def test_predict_class_with_nonexistent_file(self, mock_load_model):
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.2, 0.7]])]
        class_labels = ["Class1", "Class2", "Class3"]
        mock_load_model.return_value = (mock_session, class_labels, "input")

        with pytest.raises(FileNotFoundError):
            predict_class("nonexistent_image.jpg")


class TestResizeImage:

    def test_resize_image_correct_dimensions(self, sample_image, output_path):
        target_size = (50, 50)
        result = resize_image(sample_image, output_path, target_size)
        assert result == target_size
        assert os.path.exists(output_path)

    def test_resize_image_creates_file(self, sample_image, output_path):
        resize_image(sample_image, output_path, (75, 75))
        assert os.path.exists(output_path)

        with Image.open(output_path) as img:
            assert img.size == (75, 75)

    def test_resize_image_invalid_size(self, sample_image, output_path):
        with pytest.raises(ValueError):
            resize_image(sample_image, output_path, (0, 50))

        with pytest.raises(ValueError):
            resize_image(sample_image, output_path, (-10, 50))

    def test_resize_image_nonexistent_file(self, output_path):
        with pytest.raises(FileNotFoundError):
            resize_image("nonexistent.jpg", output_path, (50, 50))


class TestPreprocessImage:

    def test_preprocess_image_normalize(self, sample_image, output_path):
        result = preprocess_image(sample_image, output_path, normalize=True, grayscale=False)

        assert result["normalized"] is True
        assert result["grayscale"] is False
        assert os.path.exists(output_path)
        assert "original_size" in result
        assert "final_size" in result
        assert "mode" in result

    def test_preprocess_image_grayscale(self, sample_image, output_path):
        result = preprocess_image(sample_image, output_path, normalize=False, grayscale=True)

        assert result["grayscale"] is True
        assert result["mode"] == "L"
        assert os.path.exists(output_path)
