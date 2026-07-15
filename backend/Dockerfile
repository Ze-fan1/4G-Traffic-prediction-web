FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

COPY backend /app/backend
COPY Time-Series-Library-main /app/Time-Series-Library-main

ENV PYTHONPATH=/app/backend:/app/Time-Series-Library-main
ENV PORT=8000
EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
