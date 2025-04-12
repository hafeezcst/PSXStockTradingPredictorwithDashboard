FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/base.txt ./requirements-base.txt
COPY requirements/prod.txt ./requirements-prod.txt

# Install core requirements
RUN pip install --no-cache-dir -r requirements-base.txt || true
RUN pip install --no-cache-dir -r requirements-prod.txt || true

# Install streamlit specifically
RUN pip install --no-cache-dir streamlit pandas matplotlib plotly pillow fpdf2 beautifulsoup4 sqlalchemy

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs exports data/databases

# Set environment variables
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "scripts/data_processing/PSX_dashboard.py", "--server.address=0.0.0.0"]