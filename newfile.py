import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("/Users/godsentizinyon/Downloads/tf serving model/pneumonia.keras")

# Export in SavedModel format for TensorFlow Serving
model.export("/Users/godsentizinyon/Desktop/pneumonia_savedmodel")
