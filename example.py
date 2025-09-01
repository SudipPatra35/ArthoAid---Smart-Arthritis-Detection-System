from tensorflow import keras
from keras.models import load_model
# Load the full model for future prediction
model = load_model("model/best_model_1.h5")  # or "best_model.h5"

from PIL import Image
import numpy as np

def preprocess_image(path):
    img = Image.open(path).convert("L")  # Convert to grayscale
    img = img.resize((256, 256))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

# Predict
img_array = preprocess_image("test/4/9996865L.png")
pred = model.predict(img_array)
predicted_class = np.argmax(pred)

print("Predicted class:", predicted_class)

