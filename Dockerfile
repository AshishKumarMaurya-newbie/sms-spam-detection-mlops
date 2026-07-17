FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install dependencies first (to leverage Docker caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the entire application code
COPY . .

# 3. Automatically run the training pipeline to generate the model pkl artifacts
RUN python train.py

# 4. Expose the API server port
EXPOSE 8000

# 5. Spin up the application using the high-performance Uvicorn server matrix
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]