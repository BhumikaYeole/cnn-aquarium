from flask import Flask, request, jsonify
from backend.preprocess import preprocess_image
from backend.model_architecture import create_model
from flask_cors import CORS
import os
from flask import render_template
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "fish_weight.weights.h5"
    )
)

print("Starting application...")

print("BASE_DIR:", BASE_DIR)
print("WEIGHTS_PATH:", WEIGHTS_PATH)
print("WEIGHTS EXISTS:", os.path.exists(WEIGHTS_PATH))

try:
    model = create_model()
    print("Model created")

    model.load_weights(WEIGHTS_PATH)
    print("Weights loaded successfully")

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    raise


# model = load_model("../fish_classifier.h5", safe_mode=False, compile=False)

@app.route("/")
def home():
    return render_template("fish_aquarium.html")

@app.route("/predict", methods=["POST"])
def predict():   
    # print("files" , request.files)

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    # image_path = f"temp_{image_file.filename}"
    # image_file.save(image_path)

    try:
        processed_image = preprocess_image(image_file)
        # plt.imshow(processed_image[0, :, :, 0], cmap="gray")
        # plt.axis("off")

        # plt.savefig("processed.png")
        # plt.close()

        prediction = model.predict(processed_image)
        # print("Prediction:", prediction)
        if(prediction[0][0] > 0.2):
            result = "Fish"
        else:
            result = "Not Fish"
        
        return jsonify({"message": "Image processed successfully", "result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
