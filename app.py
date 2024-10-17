from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

CORS(app)
# Load the ONNX model
onnx_model_path = "Gradient_boost_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Define input names and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Endpoint to get prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request body (assume it's JSON)
        data = request.json
        
        # Convert the data into a NumPy array (9 features expected)
        sample_input = np.array([data['features']], dtype=np.float32)

        # Run inference
        output = session.run(None, {input_name: sample_input})

        # Get the prediction result
        predicted_value = output[0][0]  # Assuming the model returns one prediction
        
        # Return the result as JSON
        return jsonify({
            'predicted_gross_profit': float(predicted_value)
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

# Main driver
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
