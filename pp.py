from flask import Flask, render_template, request
import numpy as np
import librosa  # For audio processing
import numpy as np  # For numerical operations
import joblib
import numpy as np
from flask import Flask, request, render_template
app = Flask(__name__)

# Route to render the HTML form
@app.route('/')
def upload_file():
    return render_template('sample.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])

def predict():
    model = joblib.load("clf.pkl")
    # Get the uploaded file
    audio_file = request.files['file']
    features=extract_features(audio_file)
    predicted_genre=model.predict(features)
    
    return render_template('sample.html', genre=predicted_genre)
def extract_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)

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
        librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
    ]
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):  # MFCCs 1 to 20
        features.extend([np.mean(mfcc[i-1]), np.var(mfcc[i-1])])

    return np.array(features)



if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify
# import librosa  # For audio processing
# import numpy as np  # For numerical operations
# import joblib
# import numpy as np
# from flask import Flask, request, render_template

# app = Flask(__name__)
# model = joblib.load("clf.pkl")

# @app.route("/")
# def Home():
#     return render_template("index.html")
# # Route to handle prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file provided'})

#     audio_file = request.files['audio']

#     # Process the audio file
#     audio_features = extract_features(audio_file)

#     # Make prediction using your model
#     prediction = model.predict(audio_features)

#     return jsonify({'prediction': prediction})

# if __name__ == '__main__':
#     app.run(debug=True)

# # Function to process audio and make prediction
# # def extract_features(audio_file):
# #     # Load audio file
# #     y, sr = librosa.load(audio_file)

# #     # Extract features
# #     features = [
# #         librosa.get_duration(y=y, sr=sr),
# #         np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
# #         np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
# #         np.mean(librosa.feature.rms(y=y)),
# #         np.var(librosa.feature.rms(y=y)),
# #         np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
# #         np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
# #         np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
# #         np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
# #         np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
# #         np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
# #         np.mean(librosa.feature.zero_crossing_rate(y)),
# #         np.var(librosa.feature.zero_crossing_rate(y)),
# #         np.mean(librosa.effects.harmonic(y)),
# #         np.var(librosa.effects.harmonic(y)),
# #         np.mean(librosa.effects.percussive(y)),
# #         np.var(librosa.effects.percussive(y)),
# #         librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
# #     ]
    
# #     # MFCCs
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
# #     for i in range(1, 21):  # MFCCs 1 to 20
# #         features.extend([np.mean(mfcc[i-1]), np.var(mfcc[i-1])])

# #     return np.array(features)



