.PHONY: help install lint format test run train pipeline mlflow docker-build docker-run docker-push clean all

DOCKER_USERNAME ?= krypto02
IMAGE_NAME ?= mlops-lab3
DOCKER_IMAGE = $(DOCKER_USERNAME)/$(IMAGE_NAME)
PORT ?= 8000

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

lint: ## Run linting checks
	uv run flake8 logic cli api tests training
	uv run pylint logic cli api --disable=C0114,C0115,C0116,R0903,R,C

format: ## Format code with black and isort
	uv run black logic cli api tests training
	uv run isort logic cli api tests training

test: ## Run tests with coverage
	uv run pytest tests/ -v --cov=logic --cov=cli --cov=api --cov-report=term-missing --cov-report=html

train: ## Train models with MLFlow tracking
	uv run python training/train.py

pipeline: ## Run complete ML pipeline
	uv run python run_pipeline.py

mlflow: ## Start MLFlow UI
	uv run mlflow ui

run: ## Start FastAPI server
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME) .
	docker tag $(IMAGE_NAME) $(DOCKER_IMAGE):latest

docker-run: ## Run Docker container
	docker run -p $(PORT):8000 $(DOCKER_IMAGE):latest

docker-push: ## Push Docker image to Docker Hub
	docker push $(DOCKER_IMAGE):latest

clean: ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true

all: install format lint test
