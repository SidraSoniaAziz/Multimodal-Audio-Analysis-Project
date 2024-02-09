
1. Create Virtual Environment i.e, conda create -n EnvironmentName
2. Activate environment i.e, conda activate EnvironmentName
3. Install required libaries.
   1. pip install flask
   2. pip install git+https://github.com/m-bain/whisperx.git
   3. pip install tensorflow
   4. For transcription, install
      1. pip install -U openai whisper
      2. pip install git+https://github.com/openai/whisper.git
      3. pip install ffmpeg
  5. pip install numpy
3. Flask code saved in app.py file.
4. Front end code saved in files inside "templates" folder.
5. 'Static/assets' contain 'images'.
6.Gender, Age models  are inside "models" folder.
7. Uploaded audio's will be in "uploads" folder.
8. Type "python app.py" to run this project.
9. There should be one speaker in the audio. Appropriate extension for audio file is ".wav".
    Otherwise convert your audio (.mp3,.ogg,...other ) to .wav
10. Upload your audio, check gender and age.
