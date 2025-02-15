FROM python:3.12.1

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install --no-cache-dir -r ./requirements.txt

COPY ["app.py", "model.tflite", "./"]

EXPOSE 5000

CMD ["python", "app.py"]