import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
import librosa

# Create flask app
app = Flask(__name__)
model = joblib.load("clf.pkl")

@app.route("/")
def home():
    return render_template("index.html")

def extract_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Extract features
    features = [
        librosa.get_duration(y=y, sr=sr),
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.var(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.var(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.effects.harmonic(y)),
        np.var(librosa.effects.harmonic(y)),
        np.mean(librosa.effects.percussive(y)),
        np.var(librosa.effects.percussive(y)),
        tempo
    ]
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):  # MFCCs 1 to 20
        features.extend([np.mean(mfcc[i-1]), np.var(mfcc[i-1])])

    return np.array(features)

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'})

    try:
        audio_file = request.files['audio']

        # Process the audio file
        audio_features = extract_features(audio_file)

        # Reshape the features to match the input shape of the model
        audio_features = audio_features.reshape(1, -1)

        # Make prediction using your model
        dicti={0:'blues',1:'classical',2:'country',3:'disco',4:'hiphop',5:'jazz',6:'metal',7:'pop',8:'reggae',9:'rock'}
        prediction = model.predict(audio_features)
        pred=dicti[prediction[0]]
        return jsonify({'prediction_text': "The Genre is {}".format(pred)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True) 