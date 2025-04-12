# Development Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- git
- virtualenv (recommended)

## Setting Up the Development Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PSXStockTradingPredictorwithDashboard.git
cd PSXStockTradingPredictorwithDashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements/dev.txt
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "Description of your changes"
```

3. Run tests:
```bash
pytest tests/
```

4. Push your changes:
```bash
git push origin feature/your-feature-name
```

5. Create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep functions small and focused
- Write unit tests for new features

## Testing

- Write unit tests for all new features
- Run tests before committing:
```bash
pytest tests/
```

- Check test coverage:
```bash
pytest --cov=src tests/
```

## Documentation

- Update documentation when adding new features
- Keep API documentation up to date
- Add examples for new functionality
- Update README.md if necessary 