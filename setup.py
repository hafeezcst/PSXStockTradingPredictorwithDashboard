"""
Setup script for PSX Stock Trading Predictor.
"""

from setuptools import setup, find_packages

setup(
    name="psx-stock-trading-predictor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.32.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "python-dotenv>=1.0.0",
        "pytz>=2024.1",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "yfinance>=0.2.36",
        "ta>=0.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pymongo>=4.6.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.9",
        "fpdf2>=2.7.0",
        "pillow>=10.0.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "plotly-express>=0.4.1"
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.2",
            "black>=24.2.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0"
        ]
    },
    author="Muhammad Hafeez",
    author_email="your.email@example.com",
    description="PSX Stock Trading Predictor with Dashboard",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PSXStockTradingPredictorwithDashboard",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 