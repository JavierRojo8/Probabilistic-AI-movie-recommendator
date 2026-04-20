FROM python:3.11-slim

WORKDIR /app

# System deps (needed for scipy / numpy native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than the default GPU wheel)
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY src/  ./src/
COPY app/  ./app/

# Copy trained artifacts — these must exist locally before building
# Run prepare_data.py and train.py first if they are missing
COPY checkpoints/ ./checkpoints/
COPY data/processed/ ./data/processed/

# Streamlit configuration
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
