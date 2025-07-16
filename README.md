# ✍️ Handwritten Digit Recognition using Neural Networks

##  Problem Statement

The challenge is to **accurately classify handwritten digits (0 to 9)** from grayscale images. These digits vary in shape and size due to different writing styles, making this a classic problem in computer vision and pattern recognition.

---

##  Solution Overview

This project uses the **MNIST dataset** to train and evaluate a machine learning model that can recognize handwritten digits. A neural network is implemented using **TensorFlow and Keras** to perform **multi-class classification**.

---

## 🎯 Objective

To build and optimize a neural network model that can:
- Learn patterns from 28x28 grayscale digit images
- Classify digits 0 through 9 with high accuracy
- Generalize well on unseen data (test set)
- Predict digits from external handwritten images

---

## 🧰 Tools & Libraries Used

- **TensorFlow / Keras** – Deep learning framework
- **NumPy / Matplotlib** – Data processing and visualization
- **Seaborn** – For confusion matrix heatmap
- **Google Colab** – Development environment
- **PIL (Pillow)** – Image preprocessing

---

## 📚 Dataset

- **MNIST Dataset** (available in Keras)
- 60,000 training images + 10,000 test images
- Each image is 28x28 grayscale (single-channel)

---

## Approach

### 1. **Data Loading and Preprocessing**
- Loaded dataset using `keras.datasets.mnist`
- Scaled pixel values to range [0, 1]
- Flattened 28x28 images into 784-length vectors

### 2. **Model 1: Simple Neural Network**
- Architecture:
  - Input Layer: 784 nodes
  - Output Layer: 10 neurons (one per digit)
  - Activation: Sigmoid
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Epochs: 5

### 3. **Evaluation & Confusion Matrix**
- Achieved ~92.5% accuracy
- Confusion matrix showed good results but with some confusion between similar digits

### 4. **Model 2: Improved Neural Network**
- Architecture:
  - Hidden Layer: 100 neurons with `ReLU`
  - Output Layer: 10 neurons with `sigmoid`
- Epochs: 10
- Final Accuracy: ~97.7%

### 5. **Image Prediction**
- Accepted external image via upload
- Converted image to 28x28 grayscale
- Preprocessed image & predicted the digit using the trained model

---

##  Results

- **Initial Accuracy**: ~92.5%  
- **Final Accuracy** (after adding hidden layer and more epochs): **~97.7%**
- Confusion matrix shows strong performance across most digit classes

---

##  Sample Confusion Matrix

> Diagonal dominance represents correct predictions  
> Off-diagonal values represent misclassifications

<img width="1039" height="711" alt="image" src="https://github.com/user-attachments/assets/18a78341-5038-4d0e-9488-e85a624dc231" />

### **Flask Integration**
- Created `app.py` to handle image upload and prediction
- Frontend (`templates/index.html`) for user interaction
- Users can upload any digit image; it's auto-resized and processed
- Model prediction is displayed directly in the browser




---

## ⚠️ Challenges Faced

- **Poor accuracy with basic model**: Solved by adding hidden layer and switching to ReLU
- **Overfitting risk**: Controlled by using modest layer size and correct optimizer
- **Prediction on external image**: Required resizing, grayscale conversion, and reshaping to match training data format

---


## 🌐 Web Interface Demo

### 🖼️ Upload Flow
1. Visit the local web app
2. Upload a handwritten digit image
3. Get the predicted digit instantly

**Interface Highlights:**
- Clean UI with Bootstrap styling
- Drag & drop or file selection supported
- Fast TensorFlow model inference

---


## ✅ Key Learnings

- How to build and train a neural network with Keras
- Importance of data preprocessing and normalization
- Role of activation functions and hidden layers in model performance
- Visualization using confusion matrix to evaluate classification models
- Generalization of model to real-world handwritten data

---




## 📌 References

- [MNIST Dataset - Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- [Keras Documentation](https://keras.io/api/)
