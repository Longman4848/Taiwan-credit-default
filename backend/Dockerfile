FROM python:3.10-slim

WORKDIR /app


# Install system dependencies for SHAP and ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt .
COPY . .

# Expose FastAPI port
EXPOSE 8000

# For MLflow tracking
ENV MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
