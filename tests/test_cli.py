import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(tmp.name)
        yield tmp.name
    os.unlink(tmp.name)


class TestCLIPredictCommand:

    @patch("logic.image_processor.load_model")
    def test_predict_command_success(self, mock_load_model, runner, sample_image):
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.2, 0.7]])]
        class_labels = ["Abyssinian", "Bengal", "Persian"]
        mock_load_model.return_value = (mock_session, class_labels, "input")

        result = runner.invoke(app, ["predict", sample_image])
        assert result.exit_code == 0
        assert "Predicted pet breed:" in result.output

    def test_predict_command_nonexistent_file(self, runner):
        result = runner.invoke(app, ["predict", "nonexistent.jpg"])
        assert result.exit_code != 0


class TestCLIResizeCommand:

    def test_resize_command_success(self, runner, sample_image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = runner.invoke(
                app, ["resize", sample_image, output_path, "--width", "50", "--height", "50"]
            )
            assert result.exit_code == 0
            assert "Image resized to" in result.output
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_resize_command_missing_dimensions(self, runner, sample_image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        result = runner.invoke(app, ["resize", sample_image, output_path])
        assert result.exit_code != 0


class TestCLIPreprocessCommand:

    def test_preprocess_command_default(self, runner, sample_image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = runner.invoke(app, ["preprocess", sample_image, output_path])
            assert result.exit_code == 0
            assert "Image preprocessed successfully" in result.output
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preprocess_command_with_grayscale(self, runner, sample_image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = runner.invoke(app, ["preprocess", sample_image, output_path, "--grayscale"])
            assert result.exit_code == 0
            assert "Grayscale: True" in result.output
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preprocess_command_no_normalize(self, runner, sample_image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = runner.invoke(app, ["preprocess", sample_image, output_path, "--no-normalize"])
            assert result.exit_code == 0
            assert "Normalized: False" in result.output
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCLIClassesCommand:

    @patch("logic.image_processor.load_model")
    def test_classes_command(self, mock_load_model, runner):
        mock_session = MagicMock()
        class_labels = ["Abyssinian", "Bengal", "Persian", "Beagle"]
        mock_load_model.return_value = (mock_session, class_labels, "input")

        result = runner.invoke(app, ["classes"])
        assert result.exit_code == 0
        assert "Available" in result.output
        assert "Abyssinian" in result.output or "Beagle" in result.output
