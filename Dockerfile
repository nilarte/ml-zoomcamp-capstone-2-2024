FROM public.ecr.aws/lambda/python:3.10

RUN pip install flask
RUN pip install gunicorn
RUN pip install keras-image-helper
#RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install tensorflow

COPY model.tflite .
COPY predict.py .

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "predict:app"]