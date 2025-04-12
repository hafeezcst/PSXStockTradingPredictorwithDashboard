from setuptools import setup, find_packages

setup(
    name="psx-stock-predictor",
    version="1.0.0",
    description="Pakistan Stock Exchange (PSX) stock trading predictor with dashboard",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/PSX-Stock-Trading-Predictor-with-Dashboard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "psx-predictor=psx_predictor.cli:main",
        ],
    },
)