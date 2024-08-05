from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model 
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
from rapidfuzz import process, fuzz
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import logging
import psutil

app = Flask(__name__)

# Configure the SQLAlchemy part of the application instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spotyfyuser.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the UserGenres model
class UserGenres(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    genres = db.Column(db.String(500), nullable=True)

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
    try:
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
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'Error: {e}': 'Server error'}), 500

@app.route('/')
def home():
   return render_template('index.html')

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info("Application started")

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory Usage: {memory_info.rss / 1024 ** 2:.2f} MB")


# Spotify API setup
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id='33923fe14a9d46049601501e59066d27',
    client_secret='52f295db11274f6db62ef7585d7e1cd1'))

# Define features for scaling and calculations
features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
default_weights = [1/len(features)] * len(features)

# Read and preprocess the data
logging.info("Reading and preprocessing track data")
tracks_data = pd.read_csv('filtered_songs.csv')

tracks_data = tracks_data[(tracks_data['popularity'] > 40) & (tracks_data['instrumentalness'] <= 0.85)]
logging.info("Track data loaded and processed")
log_memory_usage()

#Function to fetch a song from Spotify
def get_song_from_spotify(song_name, artist_name=None):
    try:
        search_query = song_name if not artist_name else f"{song_name} artist:{artist_name}"
        logging.info(f"Searching Spotify for: {search_query}")
        results = sp.search(q=search_query, limit=1, type='track')
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            logging.info(f"Found track on Spotify: {track['name']} by {', '.join(artist['name'] for artist in track['artists'])}")
            audio_features = sp.audio_features(track['id'])[0]
            song_details = {
                'id': track['id'],
                'name': track['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': int(track['explicit']),
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'key': audio_features['key'],
                'loudness': audio_features['loudness'],
                'mode': audio_features['mode'],
                'speechiness': audio_features['speechiness'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'valence': audio_features['valence'],
                'tempo': audio_features['tempo'],
                'time_signature': audio_features['time_signature'],
            }
            return song_details
        else:
            logging.warning(f"No results found on Spotify for: {search_query}")
            return None
    except Exception as e:
        logging.error(f"Error fetching song from Spotify: {e}")
        return None

# Enhanced Fuzzy Matching Function
def enhanced_fuzzy_matching(song_name, artist_name, df):
    logging.info(f"Performing fuzzy matching for: {song_name}, {artist_name}")
    combined_query = f"{song_name} {artist_name}".strip()
    df['combined'] = df['name'] + ' ' + df['artists']
    matches = process.extractOne(combined_query, df['combined'], scorer=fuzz.token_sort_ratio)
    return df.index[df['combined'] == matches[0]].tolist()[0] if matches else None

# Function to apply the selected scaler and calculate weighted cosine similarity
def calculate_weighted_cosine_similarity(input_song_index, weights, num_songs_to_output, tracks_data, scaler_choice):
    logging.info("Calculating weighted cosine similarity")
    if scaler_choice == 'Standard Scaler':
        scaler = StandardScaler()
    else:  # MinMaxScaler
        scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(tracks_data[features]) * weights
    tracks_sparse = csr_matrix(scaled_features)
    cosine_similarities = cosine_similarity(tracks_sparse[input_song_index], tracks_sparse).flatten()
    similar_song_indices = np.argsort(-cosine_similarities)[1:num_songs_to_output+1]
    return similar_song_indices

# Function to recommend songs
def recommend_songs(song_name, artist_name, num_songs_to_output, scaler_choice, *input_weights):
    num_songs_to_output = int(num_songs_to_output)
    weights = np.array([float(weight) for weight in input_weights]) if input_weights else default_weights
    weights /= np.sum(weights)
    song_index = enhanced_fuzzy_matching(song_name, artist_name, tracks_data)
    if song_index is not None:
        similar_indices = calculate_weighted_cosine_similarity(song_index, weights, num_songs_to_output, tracks_data, scaler_choice)
        similar_songs = tracks_data.iloc[similar_indices][['name', 'artists']]
        return similar_songs, ""  # Return empty message if recommendations are found
    else:
        return pd.DataFrame(columns=['name', 'artists']), "No song found with the given details."

# Route for the main page
@app.route("/recommender", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        song_name = request.form.get("song_name")
        artist_name = request.form.get("artist_name")
        num_songs_to_output = request.form.get("num_songs_to_output", 5)
        scaler_choice = request.form.get("scaler_choice")
        weights = [request.form.get(f"weight_{feature}", 1/len(features)) for feature in features]
        
        recommendations, message = recommend_songs(song_name, artist_name, num_songs_to_output, scaler_choice, *weights)
        
        if not isinstance(recommendations, pd.DataFrame):
            recommendations = pd.DataFrame(columns=['name', 'artists'])  # Ensure recommendations is a DataFrame
            
        return render_template("rec.html", recommendations=recommendations.to_dict(orient='records'), message=message)
    
    return render_template("rec.html", recommendations=[], message="")

if __name__ == '__main__':
    app.run(debug=True)
