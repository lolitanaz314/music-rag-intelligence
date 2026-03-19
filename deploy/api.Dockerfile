FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -U pip wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "uvicorn[standard]" fastapi


EXPOSE 8000

ENV PYTHONPATH=/app


# CHANGE api.main:app if your import path differs
CMD ["bash", "-lc", "uvicorn api.main:app --host 0.0.0.0 --port 8000"]
