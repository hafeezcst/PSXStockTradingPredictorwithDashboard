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
    version="1.0.0",
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
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "plotly>=5.18.0",
        "plotly-express>=0.4.1",
        "streamlit>=1.32.0",
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
        "pytest==7.4.0",
        "black==23.12.0",
        "flake8==7.0.0",
        "mypy==1.8.0",
        "pre-commit==3.6.0",
        "fpdf2==2.7.0",
        "pillow==10.0.0",
        "PyPDF2==3.0.0",
        "python-docx==0.8.11",
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