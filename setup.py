"""
Setup script for PSX Stock Trading Predictor.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt", "r", encoding="utf-8") as fh:
    base_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("requirements/dev.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("requirements/prod.txt", "r", encoding="utf-8") as fh:
    prod_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="psx_dashboard",
    version="2.0.0",
    author="Muhammad Hafeez",
    author_email="hafeezcst@gmail.com",
    description="A comprehensive stock trading prediction and analysis tool for the Pakistan Stock Exchange (PSX)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PSXStockTradingPredictorwithDashboard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "matplotlib",
        "seaborn",
        "fpdf2",
        "yfinance",
        "beautifulsoup4",
        "sqlalchemy",
    ],
    entry_points={
        'console_scripts': [
            'psx-dashboard=scripts.data_processing.run_dashboard:main',
        ],
    },
    include_package_data=True,
    package_data={
        "psx_predictor": ["web/templates/*.html", "web/static/*"],
    },
) 