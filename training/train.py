import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

SEED = 42
DATASET_NAME = "oxford-iiit-pet"
IMAGE_SIZE = 224
EXPERIMENT_NAME = "pet-classification"
MODEL_REGISTRY_NAME = "pet-classifier"

mlflow.set_tracking_uri("file:./mlruns")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_dataset():
    print("[1/8] Downloading and preparing dataset...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.OxfordIIITPet(root=str(data_dir), download=True, transform=transform)
    print(f"[1/8] Dataset loaded: {len(full_dataset)} images")

    return full_dataset


def prepare_dataloaders(dataset, batch_size=32, train_ratio=0.8):
    print(f"[2/8] Preparing dataloaders (batch_size={batch_size})...")
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def create_model(model_name="mobilenet_v2", num_classes=37):
    print(f"[3/8] Creating {model_name} model with {num_classes} classes...")
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print("  Training...", end="", flush=True)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if batch_idx % 20 == 0:
            print(f"\r  Training... batch {batch_idx}/{len(train_loader)}", end="", flush=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def train_model(model_name="mobilenet_v2", batch_size=32, learning_rate=0.001, num_epochs=3):
    print(f"\n{'='*60}")
    print(f"Starting training: {model_name} (bs={batch_size}, lr={learning_rate})")
    print(f"{'='*60}\n")
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = download_dataset()
    num_classes = len(full_dataset.classes)
    class_labels = full_dataset.classes

    train_loader, val_loader = prepare_dataloaders(full_dataset, batch_size)

    model = create_model(model_name, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    print("[4/8] Setting up MLFlow experiment...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"{model_name}_bs{batch_size}_lr{learning_rate}"

    with mlflow.start_run(run_name=run_name):
        print(f"[5/8] Logging parameters to MLFlow (run: {run_name})...")
        mlflow.log_params(
            {
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "optimizer": "Adam",
                "criterion": "CrossEntropyLoss",
                "seed": SEED,
                "dataset": DATASET_NAME,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE,
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
            }
        )

        print("[6/8] Logging class labels artifact...")
        class_labels_dict = {"classes": class_labels}
        class_labels_path = "class_labels.json"
        with open(class_labels_path, "w") as f:
            json.dump(class_labels_dict, f)
        mlflow.log_artifact(class_labels_path)
        os.remove(class_labels_path)

        best_val_acc = 0.0

        print(f"[7/8] Training {num_epochs} epochs...\n")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        mlflow.log_metrics(
            {"final_train_acc": train_acc, "final_val_acc": val_acc, "best_val_acc": best_val_acc}
        )

        print("[8/8] Logging model to MLFlow...")
        mlflow.pytorch.log_model(model, "model", registered_model_name="pet-classifier-model")

        print(f"\nTraining completed. Best Val Acc: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MLOps Lab 3 - Training Pipeline")
    print("=" * 60)
    train_model(model_name="mobilenet_v2", batch_size=32, learning_rate=0.001, num_epochs=3)
    train_model(model_name="mobilenet_v2", batch_size=64, learning_rate=0.001, num_epochs=3)
    train_model(model_name="mobilenet_v2", batch_size=32, learning_rate=0.0001, num_epochs=3)
