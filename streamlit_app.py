import streamlit as st
import whisper
import openai

import base64
import json

openai.api_key = "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"

def summarize_text(text):
  # Use the OpenAI GPT-3 model to summarize the text
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=f"Summarize this text:\n{text}\n",
    temperature=0.7,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  # Extract the summary from the model's response
  summary = response["choices"][0]["text"]
  return summary

# Load the model
model = whisper.load_model("base")

# Set up the user interface
st.title("Transcriber")
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French", "German", "Italian"])

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")
        transcription = model.transcribe(audio_file.name, fp16=False, language=language)
        text = transcription["text"]
        st.sidebar.success("Transcription Complete")
        st.markdown(transcription["text"])
    else:
        st.sidebar.error("Please upload an audio file")

if st.sidebar.button("Summarize Text"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")
        transcription = model.transcribe(audio_file.name, fp16=False, language='English')
        text = transcription["text"]
        st.sidebar.success("Transcription Complete")
        st.markdown(transcription["text"])

        # Summarize the transcription
        summary = summarize_text(text)
        st.sidebar.success("Summarization Complete")
        print(summary)  # Debugging line
        st.markdown(summary)  # Display the summary
    else:
        st.sidebar.error("Please upload an audio file")


st.sidebar.header("Play Original Audio File")
st.sidebar.audio(audio_file)
