# CPU image for the Streamlit app and the REST API.
#   docker build -t adfd .
# Trained models are not in the image: mount a runs/ folder (see docker-compose.yml).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY app.py ./
COPY .streamlit ./.streamlit
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps -e .
ENV HF_HOME=/app/.cache/huggingface PYTHONUNBUFFERED=1
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
