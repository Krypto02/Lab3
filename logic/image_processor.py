"""Image processing and prediction using ONNX Runtime."""

import json
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image

# Model paths
ONNX_MODEL_PATH = Path("models") / "pet_classifier.onnx"
CLASS_LABELS_PATH = Path("models") / "class_labels.json"

# Cache for model and labels
_session = None
_class_labels = None
_input_name = None


def load_model():
    """Load ONNX model and class labels.

    Returns:
        tuple: (session, class_labels, input_name)
    """
    global _session, _class_labels, _input_name

    if _session is None:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        _session = ort.InferenceSession(
            str(ONNX_MODEL_PATH), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        _input_name = _session.get_inputs()[0].name

        with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
            _class_labels = labels_data["classes"]

    return _session, _class_labels, _input_name


def preprocess_image_for_model(image_path: str) -> np.ndarray:
    """Preprocess image for ONNX model inference.

    Args:
        image_path: Path to the image file

    Returns:
        np.ndarray: Preprocessed image tensor (1, 3, 224, 224)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_class(image_path: str) -> tuple:
    """Predict the class label for an image using ONNX model.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (predicted_class, confidence)
    """
    session, class_labels, input_name = load_model()

    inputs = {input_name: preprocess_image_for_model(image_path)}
    outputs = session.run(None, inputs)
    logits = outputs[0]

    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predicted_idx = np.argmax(probabilities, axis=1)[0]
    confidence = float(probabilities[0][predicted_idx])
    predicted_class = class_labels[predicted_idx]

    return predicted_class, confidence


def get_available_classes() -> List[str]:
    """Get list of available class labels from the model.

    Returns:
        List[str]: List of class labels
    """
    _, class_labels, _ = load_model()
    return class_labels


def resize_image(
    input_path: str, output_path: str, size: tuple = None, width: int = None, height: int = None
):
    """Resize an image to specified dimensions.

    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        size: Tuple of (width, height) - takes precedence
        width: Target width in pixels (used if size not provided)
        height: Target height in pixels (used if size not provided)

    Returns:
        tuple: Final size of the resized image (width, height)
    """
    img = Image.open(input_path)

    if size is not None:
        target_size = size
    elif width is not None and height is not None:
        target_size = (width, height)
    else:
        target_size = (800, 600)

    resized_img = img.resize(target_size)
    resized_img.save(output_path)

    return target_size


def preprocess_image(
    input_path: str,
    output_path: str,
    grayscale: bool = False,
    normalize: bool = True,
) -> dict:
    """Preprocess an image (legacy function).

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        grayscale: Convert to grayscale if True
        normalize: Normalize pixel values if True

    Returns:
        dict: Information about the preprocessing
    """
    img = Image.open(input_path)
    original_size = img.size

    if grayscale:
        img = img.convert("L")

    if normalize:
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    img.save(output_path)

    return {
        "original_size": original_size,
        "final_size": img.size,
        "mode": img.mode,
        "grayscale": grayscale,
        "normalized": normalize,
    }
