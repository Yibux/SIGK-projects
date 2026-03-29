FROM python:3.9-slim

WORKDIR /app

COPY 1/ /app/1/
COPY dataset/ /app/dataset/

WORKDIR /app/1

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]