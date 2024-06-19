import os
import numpy as np
import librosa
from keras.models import load_model
from flask import*

# Load the trained model from the HDF5 file
model = load_model('audio_classification.hdf5')

# Define a function to extract MFCC features from the input WAV file
def extract_mfcc_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Define a function to preprocess the extracted features
def preprocess_features(features):
    # Reshape features to match the input shape expected by the model
    return features.reshape(1, -1)

# Define a function to make predictions using the model
def predict_bird_species(wav_file):
    # Extract MFCC features from the input WAV file
    mfcc_features = extract_mfcc_features(wav_file)
    
    # Preprocess the extracted features
    preprocessed_features = preprocess_features(mfcc_features)
    
    # Make predictions using the model
    predictions = model.predict(preprocessed_features)
    
    # Get the predicted bird species
    predicted_bird_index = np.argmax(predictions)
    bird_species_mapping = {
        0: "American Robin",
        1: "Bewick's Wren",
        2: "Northern Cardinal",
        3: "Northern Mockingbird",
        4: "Song Sparrow"
    }
    
    # Get the predicted bird species name
    predicted_bird_species = bird_species_mapping.get(predicted_bird_index, "Unknown")
    
    return predicted_bird_species

app = Flask(__name__)  

UPLOAD_FOLDER='uploads'

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload',methods=['POST'])
def submit():
    if request.method == 'POST':
        f = request.files["file"]
        # if f.filename == '':
        #     return "No selected file"   
        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)
        name=predict_bird_species(file_path)  
        return render_template("output.html", name=name)


if __name__ == '__main__':
    app.run()
