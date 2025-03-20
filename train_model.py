import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import os
import cv2
import numpy as np

# Load dataset
def load_images(data_dir, label):
    images, labels = [], []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))  # Resize
        images.append(image)
        labels.append(label)
    return images, labels

# Load both classes
handloom_images, handloom_labels = load_images("dataset/handloom/", 0)
powerloom_images, powerloom_labels = load_images("dataset/powerloom/", 1)

# Convert to NumPy arrays
X = np.array(handloom_images + powerloom_images)
y = np.array(handloom_labels + powerloom_labels)

# Normalize
X = X / 255.0

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (handloom vs powerloom)
])

# Compile & Train Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train the model
model.fit(X, y, epochs=10, validation_split=0.2)

history = model.fit(X, y, epochs=10, validation_split=0.2)

# ✅ Save the trained model
model.save("saree_classifier.h5")
print("Model saved successfully as 'saree_classifier.h5' ✅")

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"✅ Training Accuracy: {train_acc:.4f}")
print(f"✅ Validation Accuracy: {val_acc:.4f}")