from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("savedmodel.pth")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    img = Image.open(request.files['file'])

    img = img.convert('L')
    img = img.resize((64,64))

    arr = np.array(img).flatten()/255.0

    prediction = model.predict([arr])[0]

    return f"Predicted Person ID: {prediction}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)