FROM python:3.10-slim

WORKDIR /app

RUN chmod +x /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD python basic_ml.py && python api.py



