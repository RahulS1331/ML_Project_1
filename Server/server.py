import io
import json
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import pywt

app = Flask(__name__)
app.template_folder = 'templates'

# Load model and class dictionary
model_path = 'C:/Users/rahul/final_saved_model.pkl'
class_dict_path = 'C:/Users/rahul/class_dictionary.json'

# Load the model using joblib
model = joblib.load(model_path)

# Load the class dictionary
with open(class_dict_path, 'r') as json_file:
    class_dict = json.load(json_file)


# Helper function to preprocess image
def preprocess_image(image):
    # Resize the image to the expected input size of the model
    image = image.resize((224, 224))
    # Convert image to array
    image_array = np.array(image)
    # Apply wavelet transform
    coeffs = pywt.wavedec2(image_array, 'db1', level=1)
    image_array = coeffs[0]  # Using only the approximation coefficients for simplicity
    # Normalize the image
    image_array = image_array / 255.0
    # Flatten the array and add batch dimension
    image_array = image_array.flatten().reshape(1, -1)
    return image_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        # Preprocess the image and make predictions
        preprocessed_image = preprocess_image(img)
        prediction_probs = model.predict_proba(preprocessed_image)[0]

        # Get class with highest probability
        top_class_index = np.argmax(prediction_probs)
        top_class = list(class_dict.values())[top_class_index]
        top_probability = float(prediction_probs[top_class_index])

        result = {"celebrity": top_class, "probability": top_probability}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
