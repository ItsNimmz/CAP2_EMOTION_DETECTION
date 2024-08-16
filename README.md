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

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ItsNimmz/CAP2_EMOTION_DETECTION.git
   cd CAP2_EMOTION_DETECTION


2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt 



3.  Set up environment variables for Spotify API keys, database configuration, and other secret keys. Ensure these are correctly set both locally and in your deployment environment.

   
4. Initialize the spotyfyuser.db database by running the provided SQLAlchemy migration scripts, ensuring all user data tables are set up properly.

5. Run the application locally

## Deployment
**To deploy the application on Render.com**:\
1.Push your code to the repository linked with your Render project.\
2.Set all environment variables correctly in the Render dashboard, including API keys and database URLs.\
3.Ensure that the spotyfyuser.db database is properly migrated in the cloud environment.\
4.Trigger a manual deployment or let Render automatically deploy on new commits.\

## Project Structure
- **app.py**:\
 Contains the main Flask application and route definitions.
- **model.py**: \
Includes the machine learning models for text and image-based mood detection.
- **filtered_songs.csv**:
 Pre-processed dataset used by the recommendation engine for better accuracy and relevance.
- **spotyfyuser.db**:\
SQLite database file for storing user profiles, preferences, and history.
- **requirements.txt**:\
 Lists all dependencies required to run the application.

We welcome contributions to improve LyreApp! \
Please fork the repository, create a feature branch, and submit a pull request with a detailed description of your changes.
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, contact at [nimisha.parameswaranthankamani@dcmail.ca / nikitha.thomas@dcmail.ca /  anju.sunny@dcmail.ca].