import json
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "pet-classification"
MODEL_NAME = "pet-classifier-model"
MODELS_DIR = Path("models")
ONNX_MODEL_PATH = MODELS_DIR / "pet_classifier.onnx"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"


def get_best_model():
    """Query registered models and select best one by validation accuracy."""
    client = MlflowClient()

    # Search model versions by name
    filter_string = f"name='{MODEL_NAME}'"
    model_versions = client.search_model_versions(filter_string=filter_string)

    if not model_versions:
        print(f"No model versions found for '{MODEL_NAME}'.")
        print("Falling back to searching runs...")

        # Fallback: search runs directly
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run training first.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.best_val_acc DESC"],
            max_results=1,
        )

        if not runs:
            raise ValueError("No runs found in experiment")

        best_run = runs[0]
        best_run_id = best_run.info.run_id
        best_val_acc = best_run.data.metrics.get("best_val_acc", 0)
        run_name = best_run.data.tags.get("mlflow.runName", "N/A")
    else:
        # Compare models by validation accuracy
        best_version = None
        best_val_acc = 0.0

        print(f"\nFound {len(model_versions)} registered model versions. Comparing...")

        for version in model_versions:
            run_id = version.run_id
            run = client.get_run(run_id)

            # Get metrics from run
            metrics = run.data.metrics
            val_acc = metrics.get("best_val_acc", 0)

            print(f"  Version {version.version}: Run {run_id[:8]}..., Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_version = version

        if best_version is None:
            raise ValueError("No valid model version found")

        best_run_id = best_version.run_id
        run = client.get_run(best_run_id)
        run_name = run.data.tags.get("mlflow.runName", "N/A")

    print("\nBest model selected:")
    print(f"  Run ID: {best_run_id}")
    print(f"  Run Name: {run_name}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")

    return best_run_id


def export_to_onnx(run_id):
    MODELS_DIR.mkdir(exist_ok=True)

    model_uri = f"runs:/{run_id}/model"
    print(f"\nLoading model from: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to ONNX: {ONNX_MODEL_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )

    print(f"Model exported successfully to {ONNX_MODEL_PATH}")

    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "class_labels.json", dst_path=str(MODELS_DIR))

    print(f"Class labels downloaded to: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        class_labels = json.load(f)

    print(f"Class labels saved: {len(class_labels['classes'])} classes")


if __name__ == "__main__":
    best_run_id = get_best_model()
    export_to_onnx(best_run_id)
    print("\nModel selection and ONNX export completed")
