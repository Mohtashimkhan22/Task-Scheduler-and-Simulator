from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Enable CORS for Next.js

flask_app = Flask(__name__)
CORS(flask_app)  # Allow cross-origin requests

# Load the trained model
model = pickle.load(open('crop_model.pkl', 'rb'))

@flask_app.route('/')
def home():
    return jsonify({"message": "Crop Prediction API is running!"})

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from Next.js
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    flask_app.run(debug=True)
