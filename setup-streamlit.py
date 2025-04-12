from setuptools import setup, find_packages

setup(
    name="psx_dashboard",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Base dependencies
        "numpy==1.24.0",
        "pandas==2.0.0",
        
        # Plotly and its dependencies
        "plotly==5.18.0",
        "plotly-express==0.4.1",
        
        # Streamlit and web dependencies
        "streamlit==1.32.0",
        "python-dotenv==1.0.0",
        
        # Data processing
        "scikit-learn==1.3.0",
        "ta==0.11.0",
        "yfinance==0.2.36",
        
        # Visualization
        "matplotlib==3.8.0",
        "seaborn==0.13.0",
        
        # Utilities
        "pytz==2024.1",
        "requests==2.31.0",
        "beautifulsoup4==4.12.0",
        "lxml==4.9.0",
        
        # Database
        "pymongo==4.6.0",
        "sqlalchemy==2.0.0",
        "psycopg2-binary==2.9.9",
        
        # Document processing
        "fpdf2==2.7.0",
        "pillow==10.0.0",
        "PyPDF2==3.0.0",
        "python-docx==0.8.11"
    ],
    python_requires=">=3.8",
) 