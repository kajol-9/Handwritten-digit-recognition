from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/digit_model.h5')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'digit' not in request.files:
        return redirect(request.url)

    file = request.files['digit']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = Image.open(filepath).convert('L').resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 784)

        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)

        return render_template('index.html', prediction=predicted_digit, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
