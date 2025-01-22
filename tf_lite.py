#!/usr/bin/env python
# coding: utf-8

import tensorflow.lite as tflite
import tensorflow as tf
from keras_image_helper import create_preprocessor

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(200, 200))

#url = 'https://raw.githubusercontent.com/nilarte/ml-zoomcamp-capstone-2-2024/refs/heads/main/data/test/young/10056.jpg'
def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    #probabilities = tf.nn.softmax(pred[0]).numpy()
    #print(probabilities)
    return pred[0]





