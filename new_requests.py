import requests
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Load and convert to grayscale
    img = Image.open(image_path).convert("L")
    # Resize to 128x128
    img = img.resize((128, 128))
    # Convert to array and normalize
    img_array = np.array(img).astype("float32") / 255.0
    # Add batch and channel dimensions -> shape (1, 128, 128, 1)
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

image_data = preprocess_image("/Users/godsentizinyon/Desktop/OsascoMLPort/TFServing/images/COVID-19 (101).jpg")

def send_rest_request(image_array, model_name="my_model"):
    url = f"http://localhost:8501/v1/models/{model_name}:predict"
    payload = {"instances": image_array.tolist()}  # Convert to serializable JSON
    response = requests.post(url=url, json=payload)
    return response.json()

result = send_rest_request(image_data)
print(result)
