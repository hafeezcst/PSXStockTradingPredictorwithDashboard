.PHONY: help install dev test lint clean build run docker-build docker-run docker-stop setup format check

help:
	@echo "Available commands:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install development dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make build      - Build the package"
	@echo "  make run        - Run the application"
	@echo "  make setup      - Setup development environment"
	@echo "  make format     - Format code"
	@echo "  make check      - Run all checks (lint, test, type)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-stop  - Stop Docker container"

install:
	pip install -r requirements/prod.txt

dev:
	pip install -r requirements/dev.txt
	pre-commit install

test:
	pytest tests/

lint:
	black --check src/ tests/
	flake8 src/ tests/
	mypy src/ tests/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

check: lint test

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel

run:
	python -m psx_predictor.web.app

setup:
	chmod +x scripts/setup_dev.sh
	./scripts/setup_dev.sh

docker-build:
	docker build -t psx-predictor -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up

docker-stop:
	docker-compose -f docker/docker-compose.yml down 