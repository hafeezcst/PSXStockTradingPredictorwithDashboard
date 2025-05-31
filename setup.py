"""
Setup script for PSX Stock Trading Predictor.
"""

from setuptools import setup, find_packages

setup(
    name="psx-stock-predictor",
    version="1.0.0",
    description="Pakistan Stock Exchange (PSX) stock trading predictor with dashboard",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.6.0",
        "sqlalchemy>=1.4.0",
        "requests>=2.26.0",
        "python-telegram-bot>=13.7",
        "python-dotenv>=0.19.0",
        "tabulate>=0.8.9",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
) 