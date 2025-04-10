import tensorflow as tf
import numpy as np

# Check TensorFlow version
print(f"Using TensorFlow version: {tf.__version__}")

# Load the trained model
model = tf.keras.models.load_model("model.h5")  # Ensure you have this file

# Convert to TensorFlow's SavedModel format if necessary
saved_model_dir = "saved_model"
model.save(saved_model_dir)

# Convert the SavedModel to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Ensure compatibility
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for performance
converter.target_spec.supported_types = [tf.float16]  # Use float16 for efficiency

tflite_model = converter.convert()

# Save the converted TFLite model
with open("saree_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model conversion successful: 'saree_classifier.tflite'")
