from flask import Flask, render_template, request, redirect, url_for
import os
import requests
import h5py
from keras.models import load_model 
from werkzeug.utils import secure_filename

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}  # Set of allowed audio file extensions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=["POST", "GET"])
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
                # return "Audio file saved successfully", 200
                print ("Audio file saved successfully")
                return render_template('options.html')
            except Exception as e:
                return f"Error saving file: {str(e)}", 500
        else:
            return "Invalid file format. Only mp3, wav, and ogg files are allowed.", 400
    return render_template('upload_form.html')  # Display the HTML form for GET requests

if __name__ == "__main__":
    app.run(debug=True)
