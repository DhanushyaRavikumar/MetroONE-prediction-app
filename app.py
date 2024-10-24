from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

CORS(app)
# Load the ONNX model
onnx_model_path = "Gradient_boost_model.onnx"
yourtrellis_onnx_model_path = "yourtrellis_model.onnx"
session = ort.InferenceSession(onnx_model_path)
session2 = ort.InferenceSession(yourtrellis_onnx_model_path)

# Define input names and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
#Session 2
input_name_2 = session2.get_inputs()[0].name
output_name_2 = session2.get_outputs()[0].name

# Print input and output names to debug
print("ONNX Model 1 input name:", input_name)
print("ONNX Model 1 output name:", output_name)

print("ONNX Model 2 input name:", input_name_2)
print("ONNX Model 2 output name:", output_name_2)

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

@app.route('/predict_yourtrellis', methods=['POST'])
def predict_yourtrellis():
    try:
        # Get input data from the request body (assume it's JSON)
        data = request.json
        
        # Convert the data into a NumPy array (9 features expected)
        sample_input = np.array([data['features']], dtype=np.float32)

        # Run inference
        output = session2.run(None, {input_name_2: sample_input})

        # Get the prediction result
        predicted_value = output[0][0]  # Assuming the model returns one prediction
        
        # Return the result as JSON
        return jsonify({
            'Amount Converted': float(predicted_value)
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

# Main driver
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
