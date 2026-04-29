FROM python:3.11-slim
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-runtime.txt
COPY src ./src
COPY models ./models
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8080"]
