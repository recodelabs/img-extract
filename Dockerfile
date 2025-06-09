FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN mkdir -p storage/images storage/logs

EXPOSE 8000

CMD ["python", "main.py"]