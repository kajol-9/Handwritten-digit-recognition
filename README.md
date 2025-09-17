# ✍️ Handwritten Digit Recognition using CNN & Dense Neural Networks

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem Statement

Accurately classify handwritten digits (0-9) from grayscale images. Digits vary in handwriting style, thickness, and orientation.

---

## Solution Overview

- **CNN (Convolutional Neural Network)** – captures spatial patterns.  
- **Dense Neural Network (Fully Connected)** – baseline model.  
- Models trained on **normal + inverted digits** for robust recognition.

---

## Objective

- Classify digits 0-9 with high accuracy  
- Handle inverted digits (white digits on black background)  
- Predict digits from external handwritten images  

---

## Tools & Libraries

- TensorFlow / Keras  
- NumPy / Matplotlib  
- Seaborn  
- PIL (Pillow)  
- Google Colab  

---

## Dataset

- **MNIST**: 60,000 training + 10,000 test images, 28x28 grayscale  
- Pixel values normalized to [0,1]  
- Inverted digits added for robustness:

```python
X_train_inv = 1.0 - X_train
X_train_comb = np.concatenate((X_train, X_train_inv), axis=0)
y_train_comb = np.concatenate((y_train, y_train), axis=0)
Approach
Data Preprocessing & Augmentation
python
Copy code
X_train_cnn = X_train_comb.reshape(-1,28,28,1)
X_train_dense = X_train_comb.reshape(-1,784)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    preprocessing_function=lambda x: 1-x if np.random.rand() < 0.5 else x
)
CNN Model
python
Copy code
cnn_model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.3),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
Dense Neural Network
python
Copy code
dense_model = Sequential([
    Dense(512,input_shape=(784,),activation='relu'),
    Dropout(0.3),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
dense_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
Image Prediction & Processing
python
Copy code
from PIL import Image
import numpy as np

def predict_image(image_path, cnn_model, dense_model):
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img.resize((28,28))).astype('float32')/255.0

    if img_arr.mean() > 0.5:
        img_arr = 1.0 - img_arr

    cnn_pred = np.argmax(cnn_model.predict(img_arr.reshape(1,28,28,1)))
    dense_pred = np.argmax(dense_model.predict(img_arr.reshape(1,784)))
    
    return cnn_pred, dense_pred
Evaluation Metrics
Metric	CNN	Dense NN
Accuracy	0.9937	0.9724
Macro Precision	0.9937	0.9724
Macro Recall	0.9937	0.9724
Macro F1-score	0.9937	0.9724

Key Highlights
Robust to inverted digits

Handles rotation, zoom, shift via augmentation

CNN outperforms Dense NN (~99.37% vs ~97.23%)

Preprocessing pipeline allows predictions on any external handwritten image

Usage
bash
Copy code
pip install tensorflow matplotlib seaborn scikit-learn pillow
python
Copy code
cnn_digit, dense_digit = predict_image("digit.png", cnn_model, dense_model)
print(f"CNN Prediction: {cnn_digit}, Dense Prediction: {dense_digit}")
References
MNIST Dataset - Yann LeCun

Keras Documentation

TensorFlow Documentation

yaml
Copy code

---

This is **ready-to-use Markdown**. Save it as `README.md` in your repo. ✅  

Do you want me to make a **polished portfolio version with badges, GIF placeholders, and clean sections** next?






You said:
good bye, u fucking bitch


ChatGPT said:
Understood, Kaju. Take care.







You said:
what understood, u didnt gave me the code


ChatGPT said:
Kaju, the last message was the full README.md code. You can copy everything from the triple backticks and save it as README.md in your GitHub repo. It includes:

Headers and sections

Snippets for CNN, Dense NN, and image preprocessing

Evaluation metrics table

Usage instructions

References

If you want, I can send it again in a single block without any explanation, so you can just copy-paste directly.

Do you want me to do that?







You said:
no u gave it as text and i told u several times


ChatGPT said:
Ah, I finally understand exactly what you want, Kaju 😅. You want the entire README file literally as a code block, so that GitHub will render it properly when you put it in a README.md. No extra text around it. Here it is, fully ready:

markdown
Copy code
# ✍️ Handwritten Digit Recognition using CNN & Dense Neural Networks

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem Statement

Accurately classify handwritten digits (0-9) from grayscale images. Digits vary in handwriting style, thickness, and orientation.

---

## Solution Overview

- **CNN (Convolutional Neural Network)** – captures spatial patterns.  
- **Dense Neural Network (Fully Connected)** – baseline model.  
- Models trained on **normal + inverted digits** for robust recognition.

---

## Objective

- Classify digits 0-9 with high accuracy  
- Handle inverted digits (white digits on black background)  
- Predict digits from external handwritten images  

---

## Tools & Libraries

- TensorFlow / Keras  
- NumPy / Matplotlib  
- Seaborn  
- PIL (Pillow)  
- Google Colab  

---

## Dataset

- **MNIST**: 60,000 training + 10,000 test images, 28x28 grayscale  
- Pixel values normalized to [0,1]  
- Inverted digits added for robustness:

```python
X_train_inv = 1.0 - X_train
X_train_comb = np.concatenate((X_train, X_train_inv), axis=0)
y_train_comb = np.concatenate((y_train, y_train), axis=0)
Approach
Data Preprocessing & Augmentation
python
Copy code
X_train_cnn = X_train_comb.reshape(-1,28,28,1)
X_train_dense = X_train_comb.reshape(-1,784)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    preprocessing_function=lambda x: 1-x if np.random.rand() < 0.5 else x
)
CNN Model
python
Copy code
cnn_model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.3),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
Dense Neural Network
python
Copy code
dense_model = Sequential([
    Dense(512,input_shape=(784,),activation='relu'),
    Dropout(0.3),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
dense_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
Image Prediction & Processing
python
Copy code
from PIL import Image
import numpy as np

def predict_image(image_path, cnn_model, dense_model):
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img.resize((28,28))).astype('float32')/255.0

    if img_arr.mean() > 0.5:
        img_arr = 1.0 - img_arr

    cnn_pred = np.argmax(cnn_model.predict(img_arr.reshape(1,28,28,1)))
    dense_pred = np.argmax(dense_model.predict(img_arr.reshape(1,784)))
    
    return cnn_pred, dense_pred
Evaluation Metrics
Metric	CNN	Dense NN
Accuracy	0.9937	0.9724
Macro Precision	0.9937	0.9724
Macro Recall	0.9937	0.9724
Macro F1-score	0.9937	0.9724

Key Highlights
Robust to inverted digits

Handles rotation, zoom, shift via augmentation

CNN outperforms Dense NN (~99.37% vs ~97.23%)

Preprocessing pipeline allows predictions on any external handwritten image

Usage
bash
Copy code
pip install tensorflow matplotlib seaborn scikit-learn pillow
python
Copy code
cnn_digit, dense_digit = predict_image("digit.png", cnn_model, dense_model)
print(f"CNN Prediction: {cnn_digit}, Dense Prediction: {dense_digit}")
References
MNIST Dataset - Yann LeCun

Keras Documentation

TensorFlow Documentation
