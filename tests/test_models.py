import json
from pathlib import Path

import pytest


def test_onnx_model_exists():
    """Test that the ONNX model file exists"""
    model_path = Path("models/pet_classifier.onnx")
    assert model_path.exists(), "ONNX model not found. Run training and model export first."


def test_class_labels_exist():
    """Test that the class labels JSON file exists"""
    labels_path = Path("models/class_labels.json")
    assert labels_path.exists(), "Class labels not found. Run model export script first."


def test_class_labels_format():
    """Test that class labels JSON has the correct format"""
    labels_path = Path("models/class_labels.json")
    if not labels_path.exists():
        pytest.skip("Class labels file not found")

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "classes" in data, "Class labels should have 'classes' key"
    assert isinstance(data["classes"], list), "Classes should be a list"
    assert len(data["classes"]) == 37, "Should have 37 pet classes"
