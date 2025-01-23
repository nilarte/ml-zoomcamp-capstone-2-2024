from flask import Flask, request, jsonify
#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
# Load the TFLite model
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor indices
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Initialize preprocessor
preprocessor = create_preprocessor('xception', target_size=(200, 200))

# Prediction function
def predict(url):
    try:
        # Preprocess the image
        X = preprocessor.from_url(url)

        # Set the input tensor
        interpreter.set_tensor(input_index, X)

        # Run inference
        interpreter.invoke()

        # Get the prediction
        pred = interpreter.get_tensor(output_index)
        return pred[0]
    except Exception as e:
        return {"error": str(e)}

# Flask route for predictions
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Parse the JSON request
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'Please provide an image URL in the request body.'}), 400

        # Perform prediction
        probabilities = predict(url)

        return jsonify({'predictions': probabilities.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
