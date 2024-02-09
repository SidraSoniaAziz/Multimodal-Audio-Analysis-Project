from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="#paste token here")

audio = "uploads/JaspreetSingh_JayShetty_2.wav"

diarization = pipeline(audio)
print(diarization)



