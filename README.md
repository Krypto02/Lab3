---
title: Pet Breed Classification
emoji: 🐾
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Pet Breed Classification

AI-powered pet breed classifier using MobileNet_v2 transfer learning with ONNX Runtime.

## Model Details

- **Architecture**: MobileNet_v2 (IMAGENET1K_V1 pretrained)
- **Dataset**: Oxford-IIIT Pet Dataset (7,349 images)
- **Classes**: 37 pet breeds (dogs and cats)
- **Validation Accuracy**: 88.59%
- **Model Size**: 8.77 MB (ONNX format)
- **Framework**: MLFlow for tracking, ONNX Runtime for inference

## Features

- Real-time breed prediction
- 37 supported breeds
- Fast inference with ONNX Runtime
- Clean and intuitive interface

## Training

Models were trained using:
- Transfer learning with frozen MobileNetV2 features
- Custom classifier head for 37 classes
- MLFlow experiment tracking
- Best model selection by validation accuracy

## Tech Stack

- **Training**: PyTorch + MLFlow
- **Inference**: ONNX Runtime
- **API**: FastAPI
- **Interface**: Gradio
- **Deployment**: Hugging Face Spaces

## Links

- [GitHub Repository](https://github.com/Krypto02/Lab3)
- [API Documentation](https://github.com/Krypto02/Lab3#api)

## Usage

Simply upload an image of a pet, and the model will predict its breed!
