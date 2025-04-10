import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Check TensorFlow version
print(f"✅ Using TensorFlow version: {tf.__version__}")

# Load dataset
def load_images(data_dir, label):
    images, labels = [], []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Skipping invalid image: {img_path}")
            continue
        image = cv2.resize(image, (128, 128))  # Resize
        images.append(image)
        labels.append(label)
    return images, labels

handloom_images, handloom_labels = load_images("dataset/handloom/", 0)
powerloom_images, powerloom_labels = load_images("dataset/powerloom/", 1)

# Combine data
X = np.array(handloom_images + powerloom_images, dtype=np.float32) / 255.0  # Normalize
y = np.array(handloom_labels + powerloom_labels, dtype=np.int32)

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Compute class weights for balancing
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2 classes: Handloom (0) and Powerloom (1)
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=100,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights_dict)

# Save Model
model.save("saree_classifier.keras")
print("✅ Model saved successfully as 'saree_classifier.keras'")

# Convert Model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for mobile
converter.target_spec.supported_types = [tf.float16]  # Reduce model size
tflite_model = converter.convert()

# Save the converted model
with open("saree_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted successfully to 'saree_classifier.tflite'")
