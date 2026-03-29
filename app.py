import pickle
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Placement Predictor API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        cgpa = float(data['cgpa'])
        iq = float(data['iq'])

        # Validation
        if cgpa < 0 or cgpa > 10:
            return jsonify({"error": "Invalid CGPA"})
        if iq < 50 or iq > 200:
            return jsonify({"error": "Invalid IQ"})

        features = np.array([[cgpa, iq]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "result": int(prediction),
            "confidence": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))