# Image-Recognition


## 🐱🐶 Image Classifier (Cat vs Dog) using Keras

This project uses a pre-trained Keras model to classify an image as either a **cat** or a **dog**.

---

## 🛠️ Requirements

* Python 3.x
* TensorFlow
* Keras
* Pillow
* NumPy

Install dependencies using:

```bash
pip install tensorflow pillow numpy
```

---

## 📁 Files

* `keras_Model.h5` → Trained Keras model
* `labels.txt` → Text file with class names (e.g., `0 Cat`, `1 Dog`)
* `image.jpg` → Test image for prediction

---

## 🔍 How It Works – Step-by-Step Explanation

```python
from keras.models import load_model      # Import Keras model loader
from PIL import Image, ImageOps          # For image processing
import numpy as np                       # For numerical operations
```

### 1. **Load the trained model and class labels**

```python
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
```

* Loads the pre-trained `.h5` model.
* Reads the class labels from `labels.txt`.

---

### 2. **Open and preprocess the image**

```python
image = Image.open("image.jpg").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
```

* Opens the image and converts it to RGB.
* Resizes it to 224×224 (the input size expected by the model).
* Normalizes pixel values to the range \[-1, 1].

---

### 3. **Prepare the input array**

```python
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array
```

* Creates an empty array with shape `(1, 224, 224, 3)` to hold one image.

---

### 4. **Run prediction**

```python
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
```

* Gets the prediction probabilities.
* Picks the index with the highest score.
* Retrieves the class name and confidence score.

---

### 5. **Display the result**

```python
print("Class:", class_name)
print("Confidence Score:", confidence_score)
```

* Prints the predicted class (Cat or Dog) and its confidence.

---

## ✅ Example Output

<img width="1227" height="126" alt="image" src="https://github.com/user-attachments/assets/928949a6-3695-4edb-9738-a307eae4ee9c" />


---

## 📌 Notes

* Make sure all files (`.h5`, `.txt`, and `.jpg`) are in the same directory.
* You can retrain the model using Teachable Machine, TensorFlow, or custom datasets.
