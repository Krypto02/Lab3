"""Gradio interface for Pet Breed Classification using Render API."""

import os
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

# Render API endpoint
RENDER_API_URL = "https://lab3-m64w.onrender.com"


def get_available_classes():
    """Get available pet breed classes from Render API."""
    try:
        response = requests.get(f"{RENDER_API_URL}/api/classes", timeout=30)
        response.raise_for_status()
        data = response.json()
        classes = data.get("classes", [])
        print(f"Fetched {len(classes)} classes from API")
        return classes
    except Exception as e:
        print(f"Error fetching classes: {e}")
        return [
            "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
            "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
            "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
            "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin",
            "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
            "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
            "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
            "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"
        ]


def predict_class(image):
    """Predict pet breed using Render API."""
    try:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(f"{RENDER_API_URL}/api/predict", files=files, timeout=30)
        response.raise_for_status()

        result = response.json()
        predicted_breed = result.get("predicted_breed", "Unknown")
        confidence = result.get("confidence", 0.0)
        return predicted_breed, confidence
    except Exception as e:
        print(f"Error predicting: {e}")
        return "Error", 0.0


css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
#header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
}
#prediction-output {
    font-size: 1.5rem;
    font-weight: bold;
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 10px;
    color: white;
    min-height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.breeds-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 1rem;
}
.breed-badge {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    text-align: center;
    font-weight: 500;
}
"""


def predict_pet_breed(image):
    """Predict pet breed from image."""
    try:
        if image is None:
            return "Please upload an image first"

        predicted_breed, confidence = predict_class(image)

        return f"**Predicted Breed:** {predicted_breed}\n\n**Confidence:** {confidence:.2%}"

    except Exception as e:
        return f"Error: {str(e)}"


def get_breeds_html():
    """Generate HTML for available breeds."""
    breeds = get_available_classes()
    if not breeds:
        return '<p style="text-align: center; padding: 1rem; color: #666;">Could not load breeds. Click Refresh Breeds button.</p>'
    badges = "".join([f'<div class="breed-badge">{breed}</div>' for breed in sorted(breeds)])
    return f'<div class="breeds-grid">{badges}</div>'


with gr.Blocks(title="Pet Breed Classifier", theme=gr.themes.Soft(), css=css) as demo:

    with gr.Column(elem_id="header"):
        gr.Markdown("# Pet Breed Classification")
        gr.Markdown("### Upload an image and let AI predict the breed")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Pet Image",
                sources=["upload", "clipboard"],
                height=400,
            )
            predict_btn = gr.Button("Predict Breed", variant="primary", size="lg", scale=1)

        with gr.Column(scale=1):
            output_text = gr.Markdown(
                "Upload an image and click predict to see results", elem_id="prediction-output"
            )

    with gr.Accordion("Supported Breeds (37 classes)", open=False):
        breeds_display = gr.HTML(value=get_breeds_html())
        refresh_btn = gr.Button("Refresh Breeds", size="sm")
        refresh_btn.click(fn=get_breeds_html, outputs=breeds_display)

    gr.Markdown(
        """
        ---
        
        **Model**: MobileNet_v2 transfer learning with ONNX Runtime  
        **Dataset**: Oxford-IIIT Pet Dataset (7,349 images)  
        **Accuracy**: 88.59% validation accuracy  
        **Framework**: MLFlow + ONNX + FastAPI  
        **API**: [https://lab3-m64w.onrender.com](https://lab3-m64w.onrender.com)  
        **Source Code**: [GitHub](https://github.com/Krypto02/Lab3)
        """
    )

    predict_btn.click(fn=predict_pet_breed, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
