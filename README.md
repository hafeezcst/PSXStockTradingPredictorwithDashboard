# PSX Stock Trading Predictor

A machine learning-based stock price prediction system for the Pakistan Stock Exchange (PSX) with a web-based dashboard.

## Features

- Historical stock data analysis
- Machine learning-based price prediction
- Technical indicator calculation
- Web-based dashboard for visualization
- REST API for data access
- Docker support for easy deployment

## Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- Docker (optional)
- Xcode Command Line Tools (for macOS users)

## Installation

### Using Conda (Recommended)

1. Create a new conda environment:
```bash
conda create -n psx-predictor python=3.8
conda activate psx-predictor
```

2. Install dependencies:
```bash
# For development
make dev

# For production only
make install
```

### Using pip

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
# For development
make dev

# For production only
make install
```

## Development Setup

1. Install development dependencies:
```bash
make dev
```

2. Run tests:
```bash
make test
```

3. Run linters:
```bash
make lint
```

4. Start the development server:
```bash
make run
```

### Streamlit Performance

For better performance with Streamlit, install the Watchdog module:
```bash
# On macOS
xcode-select --install
pip install watchdog
```

## Docker Deployment

1. Build the Docker image:
```bash
make docker-build
```

2. Run the container:
```bash
make docker-run
```

3. Stop the container:
```bash
make docker-stop
```

## Project Structure

```
psx-predictor/
├── config/             # Configuration files
├── data/              # Data storage
├── docs/              # Documentation
├── src/               # Source code
│   └── psx_predictor/
│       ├── data/      # Data handling
│       ├── models/    # ML models
│       ├── utils/     # Utilities
│       └── web/       # Web interface
├── tests/             # Test files
├── docker/            # Docker configuration
├── requirements/      # Dependencies
└── scripts/           # Utility scripts
```

## API Documentation

See [API Documentation](docs/api/README.md) for detailed API endpoints and usage.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 