# 🧠 Saree Classifier - Python Model Training

This repository contains the Python code for training a **Convolutional Neural Network (CNN)** to classify saree images as either **Handloom** or **Powerloom** using TensorFlow and Keras. The trained model is exported to TensorFlow Lite format for use in mobile applications.

---

## 📂 Dataset Structure


- Images are resized to `128x128`
- Labels: `0` for Handloom, `1` for Powerloom

---

## ⚙️ Model Training Pipeline

1. **Data Loading & Preprocessing**
   - Read and resize images using OpenCV
   - Normalize pixel values
   - Split into training and testing sets
   - Compute class weights for imbalance handling

2. **Data Augmentation**
   - Random rotations, zooms, shifts, and flips
   - Implemented using `ImageDataGenerator`

3. **CNN Model Architecture**
   - 3 Conv2D + MaxPooling layers
   - Flatten + Dense layer with Dropout
   - Output layer with 2 classes and Softmax activation

4. **Training**
   - Optimizer: `Adam`
   - Loss: `Sparse Categorical Crossentropy`
   - Accuracy metrics tracked

---

##  Run the Training

python train_model.py

5. **Dependencies**

  Install the required packages:
  pip install tensorflow opencv-python scikit-learn numpy

6.  **Files**
train_model.py – Main script for model training and TFLite conversion

saree_classifier.keras – Saved Keras model

saree_classifier.tflite – Optimized TFLite model for mobile inference

7. **📄 License**

This project is licensed under the MIT License.
