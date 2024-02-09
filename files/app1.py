import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
import h5py
import numpy as np
import whisper
# ***************Gender*****************
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding",
                              use_auth_token="#paste token here")
from pyannote.audio import Inference
inference = Inference(model, window="whole")
from tensorflow.keras.models import load_model
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
#***********Transcription**************
# whisper_model = whisper.load_model("base")
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}  # Set of allowed audio file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=["POST", "GET"])
@app.route('/options', methods=["POST", "GET"])
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
                embedding = inference(file_path)
                # transcription = whisper_model.transcribe(file_path)
                embedding = embedding.reshape(1, 1,512)
                #  print ("Audio file saved successfully")
                # return redirect(url_for('check_gender', file_path=file_path))
                gender_embedding = gender_model.predict(embedding)
                gender_result = np.argmax(gender_embedding, axis=1)
                gender_result = gender_result.max()
                age_embedding = age_model.predict(embedding)
                age_label = np.argmax(age_embedding,axis=1)
                predicted_class = [classes[label] for label in age_label]
                # transcription_result = transcription["text"]
                for label in predicted_class:
                    if gender_result == 0:
                        return render_template("options.html",gender_result = "Female",age_class=label)
                    else:
                        return render_template("options.html",gender_result = "Male",age_class=label)
                # return render_template('options.html')
            except Exception as e:
                return f"Error saving file: {str(e)}", 500
        else:
            return "Invalid file format. Only mp3, wav, and ogg files are allowed.", 400
    return render_template('upload_form.html')  # Display the HTML form for GET requests
if __name__ == "__main__":
    app.run(debug=True)
