from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model
try:
    model = joblib.load('../model/best_model.pkl')
except Exception as e:
    print("Error loading model:", str(e))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        inputs = np.array(data['inputs']).reshape(1, -1)
        print("Inputs received:", inputs)

        prediction = model.predict(inputs)[0]
        print("Raw prediction:", prediction)

        prediction = int(prediction)

        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
