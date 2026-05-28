from flask import Flask, request, jsonify
from preprocess import preprocess_image
from model_architecture import create_model
# import tensorflow as tf
# from tensorflow.keras.models import load_model
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5500",
                "http://127.0.0.1:5500"
            ]
        }
    }
)

model = create_model()
model.load_weights("../models/fish_weight.weights.h5")
# model = load_model("../fish_classifier.h5", safe_mode=False, compile=False)

@app.route("/")
def home():
    return "Fish Classifier API with CNN model"

@app.route("/predict", methods=["POST"])
def predict():   
    # print("files" , request.files)

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_path = f"temp_{image_file.filename}"
    image_file.save(image_path)

    try:
        processed_image = preprocess_image(image_path)
        plt.imshow(processed_image[0, :, :, 0], cmap="gray")
        plt.axis("off")

        plt.savefig("processed.png")
        plt.close()
        

        prediction = model.predict(processed_image)
        print("Prediction:", prediction)
        if(prediction[0][0] > 0.5):
            result = "Fish"
        else:
            result = "Not Fish"
        
        return jsonify({"message": "Image processed successfully", "result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
