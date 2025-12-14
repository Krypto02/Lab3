import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\nError: {description} failed")
        sys.exit(1)

    print(f"\n{description} completed successfully")


if __name__ == "__main__":
    print("Lab 3: Complete ML Pipeline")
    print("=" * 60)

    run_command("python training/train.py", "Training models with MLFlow tracking")

    run_command("python training/select_model.py", "Selecting best model and exporting to ONNX")

    print("\n" + "=" * 60)
    print("Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the model: pytest tests/")
    print("2. Start API server: uvicorn api.main:app --reload")
    print("3. View MLFlow experiments: mlflow ui")
