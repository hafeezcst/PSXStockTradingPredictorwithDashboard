# Contributing to PSX Stock Trading Predictor

Thank you for your interest in contributing to the PSX Stock Trading Predictor project! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Be patient and welcoming
- Be thoughtful
- Be collaborative
- Ask for help when unsure
- Stay on topic
- Be open to constructive feedback

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

If you have an idea for a new feature or enhancement, please create an issue with the following information:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or mockups
- Potential implementation approach

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request with a clear description of the changes

## Development Setup

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

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
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

## Release Process

1. Update version numbers in `setup.py` and `src/psx_predictor/__init__.py`
2. Update CHANGELOG.md with a summary of changes
3. Create a new release on GitHub
4. Tag the release with the version number

## Questions?

If you have any questions or need help, please open an issue or contact the maintainers directly.

Thank you for contributing to PSX Stock Trading Predictor!
