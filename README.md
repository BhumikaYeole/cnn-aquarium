# AI-Powered Interactive Aquarium

An interactive web application that transforms user-drawn fish doodles into animated fish swimming inside a virtual aquarium.

The application uses a Convolutional Neural Network (CNN) trained on fish doodles to determine whether a user's drawing resembles a fish before adding it to the aquarium.

---

## Features

-  Draw fish directly on a canvas
-  CNN-based fish doodle classification
-  Animated fish swimming in a virtual aquarium
-  Rejects non-fish drawings
-  Real-time prediction using Flask API
-  Custom image preprocessing pipeline
-  Deployable as a single Flask application

---

##  Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Canvas API

### Backend
- Flask
- Flask-CORS

### Machine Learning
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- NumPy
- Pillow

---

##  Project Structure

```text
fish_aquarium/
│
├── backend/
│   ├── app.py
│   ├── preprocess.py
│   ├── model_architecture.py
│   ├── requirements.txt
│   ├── templates/
│   │   └── fish_aquarium.html
│
├── models/
│   └── fish_weight.weights.h5
│
├── notebook/
│   └── FishClassifier.ipynb
│
├── failures/
│
├── README.md
└── .gitignore
```

---

##  Model Architecture

The CNN consists of:

```python
Conv2D(32) + ReLU
MaxPooling2D

Conv2D(64) + ReLU
MaxPooling2D

Flatten

Dense(128) + ReLU
Dropout(0.5)

Dense(1) + Sigmoid
```

The model outputs a probability indicating whether the drawing represents a fish.

---

##  Classification Pipeline

1. User draws on the canvas.
2. Drawing is converted into a binary image:
   - Black background
   - White fish strokes
3. Image is sent to the Flask API.
4. Backend preprocessing:
   - Grayscale conversion
   - Noise removal
   - Bounding box extraction
   - Resize to 28×28
   - Normalization
5. CNN predicts whether the doodle is a fish.
6. If classified as a fish, it is added to the aquarium.

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/BhumikaYeole/cnn-aquarium.git
cd cnn-aquarium
```

### Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Run Application

```bash
cd backend
python app.py
```

Open:

```text
http://localhost:5000
```

---

##  API Endpoint

### Predict Fish Drawing

**POST**

```http
/predict
```

**Form Data**

| Key | Type |
|------|------|
| image | File |

**Response**

```json
{
    "message": "Image processed successfully",
    "result": "Fish"
}
```

or

```json
{
    "message": "Image processed successfully",
    "result": "Not Fish"
}
```
