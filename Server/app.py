import io
import json
import joblib
from flask import Flask, request, jsonify, render_template, url_for
from PIL import Image
import numpy as np
import cv2
import pywt

app = Flask(__name__, static_url_path='/static')
app.template_folder = 'templates'

# Load model and class dictionary
model_path = 'C:/projects/Celebrity Classifier/Model/final_saved_model.pkl'
class_dict_path = 'C:/projects/Celebrity Classifier/Model/class_dictionary.json'

# Load the model using joblib
model = joblib.load(model_path)

# Load the class dictionary
with open(class_dict_path, 'r') as json_file:
    class_dict = json.load(json_file)

# Helper functions from the project code

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

def preprocess_image(image):
    # Ensure the image is in BGR format as OpenCV expects
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the cropped image if it has two eyes
    cropped_image = get_cropped_image_if_2_eyes(image)
    if cropped_image is None:
        return None

    # Resize the image to 32x32
    scaled_raw_img = cv2.resize(cropped_image, (32, 32))

    # Apply wavelet transformation
    img_har = w2d(cropped_image, 'db1', 5)
    scaled_img_har = cv2.resize(img_har, (32, 32))

    # Flatten the images and stack them
    scaled_raw_img_flat = scaled_raw_img.reshape(32 * 32 * 3, 1)
    scaled_img_har_flat = scaled_img_har.reshape(32 * 32, 1)
    combined_img = np.vstack((scaled_raw_img_flat, scaled_img_har_flat)).flatten().reshape(1, -1)

    return combined_img

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
        preprocessed_image = preprocess_image(img)
        if preprocessed_image is None:
            return jsonify({"error": "Unable to detect face with two eyes"}), 400

        prediction = model.predict(preprocessed_image)
        predicted_class = list(class_dict.keys())[list(class_dict.values()).index(prediction[0])]

        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
