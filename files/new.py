import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
import glob
import pickle
import traceback
import numpy as np
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copy2
import transcript
import h5py
from tensorflow.keras.models import load_model
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="#paste token here")

from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding",
                              use_auth_token="#paste token here")
from pyannote.audio import Inference
inference = Inference(model, window="whole")

gender_model_path = "models/gender_model.hdf5"
# Open the .hdf5 file
h5_file = h5py.File(gender_model_path, 'r')
# Load the Gender model architecture and weights
gender_model = load_model(h5_file)
# *****************Age******************
age_model_path = 'models/age_model.hdf5'
age_h5_file = h5py.File(age_model_path, 'r')
age_model = load_model(age_h5_file)
classes = {
    7:"twenties",
    6:"thirties",
    2:"fourties",
    1:"fifties",
    5:"teens",
    4:"sixties",
    3:"seventies",
    0:"eighties"
}
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}  # Set of allowed audio file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=["POST", "GET"])
@app.route('/result', methods=["POST", "GET"])
def options():
    if request.method == "POST":
        if 'audio' not in request.files:
            return "No audio", 400
        file = request.files['audio']
        if file.filename == "":
            return "No selected audio file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            folder_path = "uploads"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, filename)
            try:
                file.save(file_path)
                print("file path ", file_path)
                # Check if embeddings and mapping already exist
                if os.path.exists('embeddings.npy') and os.path.exists('mapping.pkl'):
                    embeddings = np.load('embeddings.npy', allow_pickle=True).tolist()
                    with open('mapping.pkl', 'rb') as f:
                        mapping = pickle.load(f)
                else:
                    embeddings = []
                    mapping = []
                # Get a list of all .wav files
                all_files = glob.glob("uploads/*.wav")
                for i in range(len(all_files)):
                    print(all_files[i])
                print(f"Found {len(all_files)} files to process.")
                # Function to create a new directory if it doesn't exist
                def create_dir_if_not_exists(directory):
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                # Target directory to save the combined WAV files
                target_combined_wav_dir = "combined_wav/"
                create_dir_if_not_exists(target_combined_wav_dir)

                # Target directory to save the embeddings.npy and mapping.pkl files
                target_embeddings_dir = "embeddings_mapping/"
                create_dir_if_not_exists(target_embeddings_dir)
                # Loop through all files in the directory
                for idx, filename in enumerate(all_files):
                    try:
                        print(f"Processing file {idx+1}/{len(all_files)}: {filename}")

                        # Skip if this file has already been processed
                        if any(map.get('file') == filename for map in mapping):
                            print(f"File {filename} has already been processed, skipping.")
                            continue
                        # Process the file
                        diarization = pipeline(filename, num_speakers=2)
                        audio = AudioSegment.from_wav(filename)
                        speaker_segments = {}
                        for segment, _, label in diarization.itertracks(yield_label=True):
                            start_time = int(segment.start * 1000)
                            end_time = int(segment.end * 1000)
                            audio_segment = audio[start_time:end_time]

                            if label in speaker_segments:
                                speaker_segments[label].append(audio_segment)
                            else:
                                speaker_segments[label] = [audio_segment]
                        # Combine each speaker's segments and save them to separate WAV files
                        for label, segments in speaker_segments.items():
                            combined_segment = sum(segments)
                            # Skip short segments
                            if len(combined_segment) < 1000:  # less than one second
                                continue

                            # Export the combined segment to a new .wav file in the diarized directory
                            new_filename = f"{os.path.splitext(os.path.basename(filename))[0]}speaker{label}.wav"
                            new_filepath = os.path.join(target_combined_wav_dir, new_filename)
                            combined_segment.export(new_filepath, format="wav")
                            print(f"Saved combined audio segment for speaker {label} in file {new_filepath}.")
                            embedding = inference(new_filepath)

                            # Check the dimension of the embedding and reshape if necessary
                            if len(embedding.shape) == 1:
                                embedding = embedding.reshape(1, -1)
                            elif len(embedding.shape) > 2:
                                print(f"Unexpected embedding shape in file {new_filepath}, speaker {label}. Skipping this embedding.")
                                continue
                            # Add the embedding to the list and add a mapping
                            embeddings.append(embedding)
                            embedding_index = len(embeddings) - 1  # The index of the current embedding

                            # Check if there's already a mapping for this file and speaker
                            existing_mapping = [map for map in mapping if map.get('file') == filename and map.get('speaker') == label]
                            if existing_mapping:
                                existing_mapping[0]['embedding_indices'].append(embedding_index)
                            else:
                                mapping.append({'file': filename, 'speaker': label, 'embedding_indices': [embedding_index]})
                        # Convert list of embeddings to numpy array
                        embeddings_array = np.concatenate(embeddings, axis=0)

                        # Save embeddings to a binary file in NumPy `.npy` format
                        np.save(os.path.join(target_embeddings_dir, 'embeddings.npy'), embeddings_array)
                        print(f"Saved embeddings for file {filename}.")  
                        # Save mapping to a file using pickle
                        with open(os.path.join(target_embeddings_dir, 'mapping.pkl'), 'wb') as f:
                            pickle.dump(mapping, f)
                        print(f"Saved mapping for file {filename}.")   
                    except Exception as e:
                        print(f"Encountered error with file {filename}: {e}")
                        traceback.print_exc()  # This will print the full traceback of the error

                    print("All files processed successfully.")
                
                
                # return render_template("options.html",diarization_speakers=diarization_speakers)
                # return render_template('options.html')
                # Load embeddings and mapping
                embeddings = np.load('embeddings_mapping/embeddings.npy', allow_pickle=True)
                with open('embeddings_mapping/mapping.pkl', 'rb') as f:
                    mapping = pickle.load(f)
                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(embeddings)

                # Perform agglomerative clustering with a cosine similarity threshold
                ac = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.3)
                ac.fit(1 - similarity_matrix)
                global_speaker_ids = ac.labels_

                # Add global speaker IDs to mapping
                for i, map_dict in enumerate(mapping):
                    map_dict['global_speaker_id'] = global_speaker_ids[map_dict['embedding_indices'][0]]

                # Create a folder for each cluster and copy the corresponding speaker segment into it
                for map_dict in mapping:
                    directory = f"clusters/cluster_{map_dict['global_speaker_id']}"
                    os.makedirs(directory, exist_ok=True)
                    source_file = f"combined_wav/{os.path.splitext(os.path.basename(map_dict['file']))[0]}speaker{map_dict['speaker']}.wav"
                    destination = os.path.join(directory, os.path.basename(source_file))
                    copy2(source_file, destination)
                print("successful clusters")
                clusters_directory = "clusters"
                items = os.listdir(clusters_directory)
                # Use a list comprehension to filter only the subdirectories
                subdirectories = [item for item in items if os.path.isdir(os.path.join(clusters_directory, item))]
                # Get the count of subdirectories
                no_of_speakers = len(subdirectories)
                print("No of speakers in the audio: ", no_of_speakers)
                for folder_path, _, audio_files in os.walk(clusters_directory):
                    if audio_files:
                        for audio_file in audio_files:
                            audio_path = os.path.join(folder_path, audio_file)
                            print("audio_path: ",audio_path)
                            embedding = inference(audio_path)
                            embedding = embedding.reshape(1, 1,512)
                            gender_embedding = gender_model.predict(embedding)
                            gender_result = np.argmax(gender_embedding, axis=1)
                            gender_result = gender_result.max()
                            age_embedding = age_model.predict(embedding)
                            age_label = np.argmax(age_embedding,axis=1)
                            predicted_class = [classes[label] for label in age_label]
                            for label in predicted_class:
                                if gender_result == 0:
                                    print("Gender: Female", "Age: ",label)
                                else:
                                    print("Gender: Male", "Age: ",label)
            except Exception as e:
                return f"Error saving file: {str(e)}", 500
        else:
            return "Invalid file format. Only mp3, wav, and ogg files are allowed.", 400
    return render_template('new.html')  # Display the HTML form for GET requests

if __name__ == "__main__":
    app.run(debug=True)
