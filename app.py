import gradio as gr
import tensorflow.lite as tflite
import numpy as np
from keras_image_helper import create_preprocessor
from PIL import Image
import requests
from io import BytesIO

def load_model():
    interpreter = tflite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
preprocessor = create_preprocessor('xception', target_size=(200, 200))

def predict(image):
    image = image.convert("RGB")
    image = image.resize((200, 200))
    X = np.array(image, dtype=np.float32) / 255.0  # Normalize
    X = np.expand_dims(X, axis=0)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)[0]
    probabilities = np.exp(pred) / np.sum(np.exp(pred))  # Softmax
    labels = ['Old', 'Young']
    
    return {labels[i]: float(probabilities[i]) for i in range(len(labels))}

def process_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image, predict(image)

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Textbox(label="Enter Image URL"),
    outputs=[gr.Image(label="Uploaded Image"), gr.Label(num_top_classes=2, label="Predictions")],
    title="Age Prediction Model",
    description="Enter an image URL to predict whether the person is 'Young' or 'Old'."
)

iface.launch(server_name="0.0.0.0", server_port=5000)
