from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model 
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure the SQLAlchemy part of the application instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spotyfyuser.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the UserGenres model
class UserGenres(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    genres = db.Column(db.String(500), nullable=False)

    def __init__(self, username, genres):
        self.username = username
        self.genres = genres


# Create the database and the table
with app.app_context():
    db.create_all()

@app.route('/save-genres', methods=['POST'])
def save_genres():
    data = request.get_json()
    username = data.get('username')
    genres = data.get('genres')
    
    if username and genres:
        genres_str = ','.join(genres)  # Convert list to comma-separated string
        
        # Check if a row with the same username exists
        existing_user = UserGenres.query.filter_by(username=username).first()
        if existing_user:
            # If exists, remove the existing row
            db.session.delete(existing_user)
            db.session.commit()
        
        # Insert the new row
        new_user_genres = UserGenres(username=username, genres=genres_str)
        db.session.add(new_user_genres)
        db.session.commit()
        
        return jsonify({'message': 'Genres saved successfully'}), 200
    else:
        return jsonify({'message': 'Invalid data'}), 400

@app.route('/get-genres/<username>', methods=['GET'])
def get_genres(username):
    user_genres = UserGenres.query.filter_by(username=username).first()
    if user_genres:
        genres_list = user_genres.genres.split(',')  # Convert string back to list
        return jsonify({'username': username, 'genres': genres_list}), 200
    else:
        return jsonify({'message': 'User not found'}), 404


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
