import streamlit as st
import openai
import whisper
import constants

openai.api_key =  st.secrets["API_KEY"] or constants.API_KEY

st.set_option('deprecation.showfileUploaderEncoding', False)

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

  # Update the progress bar

  return summary

# Set up the user interface
st.title("Summarizer")
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
if audio_file is not None:
  audio_progress = st.progress(0)
  audio_file.seek(0, 2)
  total_size = audio_file.tell()
  audio_file.seek(0)
  uploaded_size = 0
  while uploaded_size < total_size:
    chunk = audio_file.read(1024)
    uploaded_size += len(chunk)
    audio_progress.progress(int((uploaded_size / total_size) * 100))
  
  # Summarize Audio
  if st.button("Summarize Audio"):
    if audio_file is not None:
        with st.spinner("Transcribing audio..."):
          # Load the model and transcribe the audio
          model = whisper.load_model("base")
          transcription = model.transcribe(audio_file.name, fp16=False, language='English')
          text = transcription["text"]
        
        st.success("Transcription Complete")
        transcription_text_area = st.text_area("", value=transcription["text"])
        st.download_button('Download Transcript', text,"transcript.txt")

        # Show the spinner again
        with st.spinner("Summarizing text..."):
          # Summarize the transcription
          summary = summarize_text(text)

        st.success("Summarization Complete")
        summary_text_area = st.text_area("", value=summary)  # Display the summary
        

        st.download_button('Download Summary', summary,"summary.txt")

        
    else:
        st.error("Please upload an audio file")

text_file = st.file_uploader("Upload Text", type=["txt"])
if text_file is not None:
  text_progress = st.progress(0)
  text_file.seek(0, 2)
  total_size = text_file.tell()
  text_file.seek(0)
  uploaded_size = 0
  while uploaded_size < total_size:
    chunk = text_file.read(1024)
    uploaded_size += len(chunk)
    text_progress.progress(int((uploaded_size / total_size) * 100))

  # Summarize Text
  if st.button("Summarize Text"):
    if text_file is not None:
        # Show the spinner again
        with st.spinner("Summarizing text..."):
          # Read the contents of the file into a string
          file_contents = text_file.read().decode()
          
          # Pass the file contents as a string to the open() function
          with open(text_file.name, 'r',encoding='utf8') as file:
              text = file.read()
              summary = summarize_text(text)
              st.success("Summarization Complete")
              summary_text_area = st.text_area("", value=summary)  # Display the summary
              
              st.download_button('Download Summary', summary,"summary.txt")

    else:
        st.error("Please upload a text file")

st.sidebar.header("Play Original Audio File")
st.sidebar.audio(audio_file)
