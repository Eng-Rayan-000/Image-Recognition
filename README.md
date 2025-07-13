

---

# 🐶🐱 Cat vs Dog Image Classifier using Keras

This project is a simple deep learning model that classifies an input image as either a **cat** or a **dog** using a pre-trained Keras model.

---

## 📁 Project Files

| File Name        | Description                                      |
| ---------------- | ------------------------------------------------ |
| `keras_model.h5` | Pre-trained Keras model for image classification |
| `labels.txt`     | Class labels file (e.g., `0 Cat`, `1 Dog`)       |
| `image.jpg`      | Sample image to classify                         |
| `main.py`        | Python script that loads the model and predicts  |

---

## 🛠 Requirements

Make sure you have the following Python packages installed:

```bash
pip install tensorflow pillow numpy
```

---

## 📸 How It Works – Step-by-Step

### 1. **Import required libraries**

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
```

* Loads the model from disk.
* Uses Pillow to process the input image.
* Uses NumPy for numerical operations.

---

### 2. **Load the trained model and class labels**

```python
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
```

* `keras_model.h5` is a trained Keras model (e.g., from Teachable Machine).
* `labels.txt` contains the class names.

---

### 3. **Prepare the image for prediction**

```python
image = Image.open("image.jpg").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
```

* Opens the image and ensures it's in RGB format.
* Resizes the image to 224×224 pixels.
* Normalizes the pixel values to the range \[-1, 1].

---

### 4. **Create input array and run prediction**

```python
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
```

* Sets up the data in the shape the model expects.
* Predicts the class.
* Extracts the predicted class name and confidence.

---

### 5. **Display the result**

```python
print("Class:", class_name)
print("Confidence Score:", confidence_score)
```

* Outputs the classification result and how confident the model is.

---

## ✅ Output


<img width="1227" height="126" alt="image" src="https://github.com/user-attachments/assets/7040b940-f157-4a31-a17e-a07ccf2e4052" />






