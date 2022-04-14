from flask import Flask, jsonify, request
from ProjectClassifier import get_prediction

app = Flask(__name__)

@app.route("/")
def hello():
    return "Alphabet Recognition"

@app.route("/predict-alphabet", methods=["POST"])
def predict_data():
    image = request.files.get("alphabet")
    prediction = get_prediction(image)
    return jsonify({
        "prediction": prediction
    }), 200

if __name__ == "__main__":
    app.run(debug=True)