name: MNIST Inverted Digit Classification
description: >
  This project implements a deep learning pipeline to classify handwritten digits from the MNIST dataset, including inverted digits. 
  Two models are trained: a Convolutional Neural Network (CNN) and a Dense Neural Network (Dense NN). 
  It also supports predicting digits from custom images using automatic preprocessing.

**features**:
  - Load and normalize MNIST dataset
  - Generate inverted images to improve robustness
  - Shuffle combined dataset for training and testing
  - Train CNN and Dense NN models with dropout regularization
  - Apply data augmentation: rotation, shift, zoom, and random inversion
  - Evaluate models using accuracy, confusion matrix, precision, recall, and F1-score
  - Predict digits from custom images with automatic resizing, normalization, and inversion handling

**requirements**:
  python: ">=3.9"
  packages:
    - tensorflow
    - matplotlib
    - seaborn
    - scikit-learn
    - pillow
    - google-colab

**installation**:
  - pip install tensorflow matplotlib seaborn scikit-learn pillow

**usage**:
  description: >
    Run this notebook in Google Colab or locally to train CNN and Dense NN models on the MNIST dataset with inverted digits.
  steps:
    - Load and normalize MNIST dataset
    - Generate inverted images
    - Shuffle and reshape dataset for CNN and Dense NN
    - Apply data augmentation
    - Train CNN model
    - Train Dense NN model
    - Evaluate models and visualize confusion matrix
    - Upload custom images to predict digits

**data_preprocessing**:
  normalization: "Pixel values scaled to [0,1]"
  inversion: "1.0 - X_train / X_test"
  reshaping:
    cnn: "(-1,28,28,1)"
    dense: "(-1,28*28)"
  augmentation:
    rotation_range: 10
    zoom_range: 0.1
    width_shift_range: 0.1
    height_shift_range: 0.1
    random_inversion: true

**cnn_model**:
  architecture:
    - Conv2D: 32 filters, 3x3, ReLU
    - Conv2D: 64 filters, 3x3, ReLU
    - MaxPooling2D: 2x2
    - Dropout: 0.25
    - Flatten
    - Dense: 256, ReLU
    - Dropout: 0.3
    - Dense: 128, ReLU
    - Dropout: 0.3
    - Dense: 64, ReLU
    - Dense: 10, Softmax
  optimizer: Adam
  loss: sparse_categorical_crossentropy
  metrics: [accuracy]

**dense_model**:
  architecture:
    - Dense: 512, ReLU
    - Dropout: 0.3
    - Dense: 256, ReLU
    - Dropout: 0.3
    - Dense: 128, ReLU
    - Dense: 10, Softmax
  optimizer: Adam
  loss: sparse_categorical_crossentropy
  metrics: [accuracy]

**evaluation**:
  metrics:
    macro_precision: 0.9937
    macro_recall: 0.9937
    macro_f1_score: 0.9937
    cnn_accuracy: 0.9937
    dense_nn_accuracy: 0.97235
  confusion_matrix: true
  classification_report: true

**predict_custom_image**:
  description: >
    Upload a handwritten digit image (PNG/JPG). The system automatically resizes it to 28x28 pixels, converts to grayscale, normalizes pixel values, and inverts colors if necessary (white background detected). This ensures the models can correctly classify the digit regardless of input format.
  steps:
    - Load image using PIL
    - Convert to grayscale
    - Resize to 28x28
    - Convert to numpy array and normalize
    - Auto-invert if mean pixel value > 127
    - Reshape for CNN (1,28,28,1) or Dense NN (1,784)
    - Predict digit using both models
  example:
    uploaded = files.upload()
    image_path = next(iter(uploaded.keys()))
    cnn_digit, dense_digit = predict_image(image_path, cnn_model, dense_model)
    print(f"Predicted Digit (CNN): {cnn_digit}")
    print(f"Predicted Digit (Dense NN): {dense_digit}")

**image_processing_background**:
  description: 
    The image processing pipeline ensures that any custom input image is compatible with the trained models:
      1. Grayscale conversion: converts color images to single-channel intensity images.
      2. Resizing: scales image to 28x28 pixels.
      3. Normalization: pixel values scaled to [0,1] to match training data.
      4. Auto-inversion: if the background is white, invert pixel values so that digits are dark on light background.
      5. Reshape: converts image into a format suitable for CNN (28x28x1) or Dense NN (784).
    These steps guarantee high prediction accuracy even on unseen and non-standard images.

**model_saving**:
  cnn_model: cnn_model_inverted.keras
  dense_model: dense_model_inverted.keras


  github: https://github.com/kajol-9
  email: kajol@example.com
