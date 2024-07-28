import joblib
import json
import numpy
import base64
import cv2
from wavelet import w2d
def classify_image(image_base64_data,file_path=None):
    pass


def get_cropped_if_2_eyes(image_path,image_base64_data):
    face_detector1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # variables made with fn
    eye_detector1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img = cv2.imread(image_path)
    if img is None:
        with open('error_log.txt', 'a') as f:
            f.write(f"Error: Unable to load image at {image_path}\n")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_result = face_detector1.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces_result:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_detector1.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None


def get_b64_test_image_for_scarlett():
    with open("b64.txt") as f:
           return f.read()

if __name__ == "__main__" :
    print(classify_image(get_b64_test_image_for_scarlett(),None))