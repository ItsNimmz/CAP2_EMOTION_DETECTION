#  LyreApp - Music Recommendation and Emotion Detection Application
LyreApp is a sophisticated application that recommends music based on users' moods. It leverages advanced machine learning models to analyze both text inputs and detect emotions from real-time camera images, providing personalized music recommendations through Spotify integration.

## Features
- **Image-Based Mood Detection**: Uses real-time image recognition via facial expressions to assess mood. (Currently facing deployment issues).
- **Spotify Integration**: Delivers personalized music recommendations based on detected mood using Spotify’s vast library.
- **Data Management**: Efficiently handles user data using SQLAlchemy for persistent storage. The spotyfyuser.db database stores user data such as profiles and song history.
- **Filtered Song Dataset**: Utilizes a pre-processed dataset (filtered_songs.csv) to enhance the recommendation engine, ensuring more relevant and diverse music suggestions.

## Deployment

LyreApp is deployed on Render.com. Deployment has been successful, but the image recognition feature is not functioning as expected in the production environment. We are investigating the issue, which could be related to environment configuration, dependency versions, or file path errors.
Known Issues

- **Image Recognition**: Image-based mood detection is currently not working properly in the deployed environment. 

## Prerequisites

Python 3.7+\
Flask 3.0.2\
Gunicorn 19.9.0\
TensorFlow\
PyTorch 2.0.0+cu117\
OpenCV\
Pillow\
SQLAlchemy\
pandas\
scikit-learn\
scipy\
rapidfuzz\
spotipy\
psutil

All dependencies are listed in the requirements.txt file.
## Installation

**Clone the repository**:
```bash
git clone https://github.com/ItsNimmz/CAP2_EMOTION_DETECTION.git
cd CAP2_EMOTION_DETECTION

**Install the required dependencies**:
```bash
pip install -r requirements.txt
