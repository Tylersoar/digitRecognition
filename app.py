
from flask import Flask, request, jsonify
import torch
import numpy as np
from model_def import CNN
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=torch.device("cpu")))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    # converts JSON data to pytorch tensor
    data = request.get_json()
    pixels = data["pixels"]  # array of 784 floats

    image = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, 1).item()

    # returns the predicted digit as JSON
    return jsonify({"prediction": predicted})


if __name__ == "__main__":
    app.run(debug=True)
