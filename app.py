from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model 
from io import BytesIO
from PIL import Image
from flask_cors import CORS
app = Flask(__name__)


# Load the pre-trained model for facial expression recognition
model = load_model('emotion_model.h5')

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        prediction = model.predict(roi_gray)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]
        emotions.append(emotion)

    return emotions

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    print('heree..............')
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    x = request.files['image']
    image = np.frombuffer(x.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    emotions = detect_emotion(image)

    if emotions:
        return jsonify({'emotions': emotions}), 200
    else:
        return jsonify({'error': 'No faces detected'}), 400
    

@app.route('/')
def home():
   return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
