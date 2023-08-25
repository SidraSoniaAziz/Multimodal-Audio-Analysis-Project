from pydub import AudioSegment
from scipy.io import wavfile
import os
#pip3 install git+https://github.com/linto-ai/whisper-timestamped
import whisper_timestamped as whisper
import pandas as pd

#pip install transformers
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Emotion
#pip install speechbrain
##git clone https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP
#git lfs install
from speechbrain.pretrained.interfaces import foreign_class

# Open AI
#!pip install openai
import openai

import argparse

argparser = argparse.ArgumentParser(description='Process parameters')

argparser.add_argument('--device', type=str, default="cpu", help='Device to train on')
argparser.add_argument('--customer_first',  type=bool, default=True,   help='Channel 1 is customer or not')
argparser.add_argument('--openaikey', type=str,   default="xxxxx", help='openai_key')
argparser.add_argument('--audio_file', type=str,   default="test.mp3", help='Path of the audio')
argparser.add_argument('--whisper_model', type=str,   default="medium", help="Whisper model type: tiny, small, medium ...")

argparser.add_argument('--sentiment_flag',  type=bool, default=False,   help='Extract sentiment')
argparser.add_argument('--emotion_flag',  type=bool, default=False,   help='Extract emotion')
argparser.add_argument('--transcript_done',  type=bool, default=False,   help='Extract transcript or take from saved one')

args = argparser.parse_args()

def mp3towav(audio_file, dir_path):

    sound = AudioSegment.from_mp3(audio_file)
    file_split = audio_file.rsplit("/", 1)
    new_name = file_split[1].rsplit(".",1)[0] + ".wav"
    print(new_name)
    sound.export(os.path.join(dir_path, new_name), format="wav")

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def preprocess(audio_file):

    dir_path = audio_file[:-4]

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    audio = AudioSegment.from_mp3(audio_file)

    audio = audio
    mono_audios = audio.split_to_mono()

    path_dir_1 = os.path.join(dir_path, 'audio_channel_1')
    path_dir_2 = os.path.join(dir_path, 'audio_channel_2')

    if not os.path.isdir(path_dir_1):
        os.mkdir(path_dir_1)

    if not os.path.isdir(path_dir_2):
        os.mkdir(path_dir_2)

    path_file_1 = os.path.join(path_dir_1, 'audio_channel_1.mp3')
    path_file_2 = os.path.join(path_dir_2, 'audio_channel_2.mp3')

    mono_audios[0].export(path_file_1, format="mp3")
    mono_audios[1].export(path_file_2, format="mp3")

    return path_file_1, path_file_2, path_dir_1, path_dir_2, dir_path

def transcript_diarization(path_file_1, path_file_2, path_dir_1, path_dir_2):

    model = whisper.load_model(args.whisper_model, device=args.device)

    audio_1 = whisper.load_audio(path_file_1)
    result_1 = whisper.transcribe(model, audio_1, language="en")

    audio_2 = whisper.load_audio(path_file_2)
    result_2 = whisper.transcribe(model, audio_2, language="en")

    frame_1 = pd.DataFrame.from_dict(result_1['segments'])
    frame_1 = frame_1.drop(columns=['temperature','tokens','seek','id', 'avg_logprob','compression_ratio','no_speech_prob',	'confidence'])

    frame_2 = pd.DataFrame.from_dict(result_2['segments'])
    frame_2 = frame_2.drop(columns=['temperature','tokens','seek','id', 'avg_logprob','compression_ratio','no_speech_prob',	'confidence'])

    # Who talks first, customer or agent
    channel_1 = "Customer"
    channel_2 = "Agent" if channel_1=="Customer" else "Customer"

    frame_1.to_csv(path_dir_1+"/frame_transcripts_1.csv", index=False)
    frame_2.to_csv(path_dir_2+"/frame_transcripts_2.csv", index=False)

    frame_1 = pd.read_csv(path_dir_1+"/frame_transcripts_1.csv")
    frame_2 = pd.read_csv(path_dir_2+"/frame_transcripts_2.csv")

    # import ast
    #
    # frame_1_words = pd.DataFrame.from_dict(frame_1.loc[0,'words']) if not type(frame_1.loc[0,'words']) == str else pd.DataFrame(list(ast.literal_eval(frame_1.loc[0,'words'])))
    #
    # for index, row in frame_1.iloc[1:].iterrows():
    #   new_row_1 = pd.DataFrame.from_dict(row["words"]) if not type(frame_1.loc[0,'words']) == str else pd.DataFrame(list(ast.literal_eval(row["words"])))
    #   frame_1_words = pd.concat([frame_1_words, new_row_1], ignore_index=True).drop(columns=[0], errors='ignore')
    #
    # frame_2_words = pd.DataFrame.from_dict(frame_2.loc[0,'words']) if not type(frame_2.loc[0,'words']) == str else pd.DataFrame(list(ast.literal_eval(frame_2.loc[0,'words'])))
    #
    # for index, row in frame_2.iloc[1:].iterrows():
    #   new_row_2 = pd.DataFrame.from_dict(row["words"]) if not type(frame_2.loc[0,'words']) == str else pd.DataFrame(list(ast.literal_eval(row["words"])))
    #   frame_2_words = pd.concat([frame_2_words, new_row_2], ignore_index=True).drop(columns=[0], errors='ignore')
    #

    frame_1_words = frame_1
    prev_row = None
    for i, row in frame_1_words.iterrows():
        if prev_row is not None and row['text'] == prev_row['text']:
            frame_1_words.drop(i, inplace=True)
        else:
            prev_row = row

    frame_2_words = frame_2
    prev_row = None
    for i, row in frame_2_words.iterrows():
        if prev_row is not None and row['text'] == prev_row['text']:
            frame_2_words.drop(i, inplace=True)
        else:
            prev_row = row

    frame_1_words = frame_1_words.drop(columns=["confidence"], errors='ignore')
    frame_1_words["speaker"] = ["Speaker1" for i in range(len(frame_1_words))]

    frame_2_words = frame_2_words.drop(columns=["confidence"], errors='ignore')
    frame_2_words["speaker"] = ["Speaker2" for i in range(len(frame_2_words))]

    frame_total_word = pd.concat([frame_1_words, frame_2_words], ignore_index = False).sort_values(by=['start', 'end'], ignore_index=True)
    frame_total = pd.DataFrame(columns=['text', 'start', 'end', 'speaker'])

    i = 0
    while i < len(frame_total_word):
        # get the current speaker
        speaker = frame_total_word.iloc[i]['speaker']
        # initialize variables for the merged row
        text = frame_total_word.iloc[i]['text']
        start = frame_total_word.iloc[i]['start']
        end = frame_total_word.iloc[i]['end']
        # loop through the rows for the current speaker
        j = i + 1
        while j < len(frame_total_word) and frame_total_word.iloc[j]['speaker'] == speaker:
            # concatenate the text and add the start and end times
            text += ' ' + frame_total_word.iloc[j]['text']
            end = frame_total_word.iloc[j]['end']
            j += 1
        # add the merged row to the new dataframe
        frame_total = pd.concat([frame_total,pd.DataFrame.from_dict({
            'text': [text],
            'start': [start],
            'end': [end],
            'speaker': [speaker]})], ignore_index=True)
        # update the counter
        i = j

    frame_total.to_csv(dir_path + "/frame_transcripts.csv", index=False)

    return frame_total

# Sentiment
def sentiment(frame_total):

    sentiment_dict = {0:"Negative", 1: "Neutral", 2: "Positive"}

    tokenizer = AutoTokenizer.from_pretrained("Souvikcmsa/BERT_sentiment_analysis") #assemblyai/distilbert-base-uncased-sst2
    model = AutoModelForSequenceClassification.from_pretrained("Souvikcmsa/BERT_sentiment_analysis")

    sentiment = []
    for index, row in frame_total.iterrows():
      tokenized_segments = tokenizer([row["text"]], return_tensors="pt", padding=True, truncation=True)
      tokenized_segments_input_ids, tokenized_segments_attention_mask = tokenized_segments.input_ids, tokenized_segments.attention_mask
      sentiment.append(sentiment_dict[np.argmax(F.softmax(model(input_ids=tokenized_segments_input_ids, attention_mask=tokenized_segments_attention_mask)['logits'], dim=1).detach().numpy())])

    frame_total["Sentiment"] = sentiment

    return frame_total

def time_speakers(frame_total):

    frame_total["speak_time"] = frame_total["end"]-frame_total["start"]
    results = frame_total.groupby("speaker")["speak_time"].sum()
    tot = results["Agent"]+results["Customer"]
    perc_agent, perc_customer = results["Agent"]/tot, results["Customer"]/tot

    if (int(perc_agent * 100) + int(perc_customer * 100)) != 100:
        out1 = int(perc_agent * 100) + 1
        out2 = int(perc_customer * 100)
    else:
        out1 = int(perc_agent * 100)
        out2 = int(perc_customer * 100)

    return out1, out2

# Emotion
def emotion(frame_total, path_file_1, path_file_2):

    emotion = []
    audio_1 = AudioSegment.from_wav(path_file_1)
    audio_2 = AudioSegment.from_wav(path_file_2)

    for index, row in frame_total.iterrows():
      start_ms = row["start"]*1000
      end_ms = row["end"]*1000

      total_ms_sec = end_ms-start_ms
      pad = total_ms_sec*0.1
      if total_ms_sec < 1000:
        emotion.append(['None'])
        continue

      if row["speaker"] == "Speaker1":
        audio = audio_1[int(start_ms+pad):int(end_ms-pad)]
      else:
        audio = audio_2[int(start_ms+pad):int(end_ms-pad)]

      audio.export('test.wav', format='wav')
      classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
      out_prob, score, index, text_lab = classifier.classify_file("test.wav")
      emotion.append(text_lab)
      os.remove("test.wav")

      print(text_lab)

    frame_total["emotion"] = [val[0] for val in emotion]
    return frame_total

def summarization(result):

    prompt_response = []
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_num = count_tokens(result)

    if token_num > 4090:

        chunks = break_up_file_to_chunks(result)
        for i, chunk in enumerate(chunks):
            prompt_request = "Summarize this conversation between customer and agent: " + tokenizer.decode(chunks[i])

            messages = [{"role": "system", "content": "This is text summarization."}]
            messages.append({"role": "user", "content": prompt_request})

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=.5,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            prompt_response.append(response["choices"][0]["message"]['content'].strip())

        prompt_request = "Consolidate these conversation summaries in two sentences: " + str(prompt_response)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                #{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_request},

            ])

    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                #{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarize this conversation between customer and agent in two sentences: " + result},

            ])

    summary = response["choices"][0]["message"]["content"]

    return summary, token_num

def credit_card(result, token_num):

    prompt_response = []
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if token_num > 4090:
        chunks = break_up_file_to_chunks(result)
        for i, chunk in enumerate(chunks):
            prompt_request = "In the following conversation, find any information related to a credit card number, if there is none, print only None:" + tokenizer.decode(chunks[i])
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                temperature=.5,
                messages=[
                    {"role": "user",
                     "content": prompt_request }])

        prompt_response.append(response["choices"][0]["message"]['content'].strip())
        prompt_response.remove("None") if "None" in prompt_response else None
        if len(prompt_response) == 0:
            prompt_response = "None"
        credit_card_info = prompt_response

    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            temperature=.5,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": "In the following conversation, find any information related to a credit card number, if there is none, just print None:" + result}])

        credit_card_info = response["choices"][0]["message"]["content"]

    return credit_card_info

def keywords(result, token_num):

    prompt_response = []
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if token_num > 4090:
        chunks = break_up_file_to_chunks(result)
        for i, chunk in enumerate(chunks):
            prompt_request = "Give me the most relevant keywords of the following conversation:" + tokenizer.decode(chunks[i])
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                temperature=.5,
                messages=[
                    {"role": "user",
                     "content": prompt_request}])
            prompt_response.append(response["choices"][0]["message"]['content'].strip())

        keywords=""
        for val in prompt_response:
            keywords += " " + val
    else:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        #{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content" : "Give me the most relevant keywords of the following conversation:" + result},
        ])
        keywords = response["choices"][0]["message"]["content"]

    return keywords

def count_tokens(text):

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_tokens = len(tokenizer.encode(text))
    return num_tokens


def break_up_file_to_chunks(text, chunk_size=2000, overlap=100):

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    chunks = []
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

if __name__ == "__main__":

    openai.api_key = args.openaikey
    os.environ["OPENAI_API_KEY"] = args.openaikey

    path_file_1, path_file_2, path_dir_1, path_dir_2, dir_path = preprocess(args.audio_file)

    if args.transcript_done:
        frame_total = pd.read_csv(dir_path + "/frame_transcripts.csv")
    else:
        frame_total = transcript_diarization(path_file_1, path_file_2, path_dir_1, path_dir_2)

    # Sentiment
    if args.sentiment_flag:
        frame_total = sentiment(frame_total)
    # Emotion
    if args.emotion_flag:
        frame_total = emotion(frame_total, path_file_1, path_file_2)

    if args.customer_first:
        frame_total["speaker"] = frame_total["speaker"].replace("Speaker1", "Customer")
        frame_total["speaker"] = frame_total["speaker"].replace("Speaker2", "Agent")
    else:
        frame_total["speaker"] = frame_total["speaker"].replace("Speaker1", "Agent")
        frame_total["speaker"] = frame_total["speaker"].replace("Speaker2", "Customer")

    frame_total = frame_total.rename(columns={'text': 'transcription'})
    cols = ['start', 'end', 'transcription', 'speaker', 'emotion', 'Sentiment']

    if not args.sentiment_flag:
        cols.remove('Sentiment')
    if not args.emotion_flag:
        cols.remove('emotion')

    # Write transcriptions inside transcript.txt
    frame_total = frame_total[cols]
    frame_total.to_csv(dir_path + "/frame_transcripts_total.csv", index=False)

    perc_agent, perc_customer = time_speakers(frame_total)

    import shutil
    shutil.rmtree(path_dir_1)
    shutil.rmtree(path_dir_2)

    file_txt = open(dir_path + "/transcript.txt", "w")
    for index, row in frame_total.iterrows():
        if (index > 0) and (frame_total.loc[index, "speaker"] == frame_total.loc[index - 1, "speaker"]):
            file_txt.write(" " + row['text'])
        else:
            if index != 0:
                file_txt.write("/n" + " ")
            file_txt.write(row['speaker'] + ":" + " ")
            file_txt.write(row["transcription"])
    file_txt.close()

    f = open(dir_path + "/transcript.txt")

    text_string = f.read()
    #text_string = text_string + text_string + text_string + text_string

    summary, token_num = summarization(text_string)

    keywords = keywords(text_string, token_num)
    keywords = keywords.replace("Keywords:", '')

    credit_card_info = credit_card(text_string, token_num)

    dictot = {"summary":[summary], "keywords":[keywords], "credit_card":[credit_card_info], "perc_talk_agent":[perc_agent], "perc_talk_customer":[perc_customer]}
    frame_more = pd.DataFrame(dictot)
    frame_more.to_csv(dir_path + "/audio_info.csv", index=False)






