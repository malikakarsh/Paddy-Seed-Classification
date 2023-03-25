FROM python:3.9.16-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip &&  \\
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "model.py"]


