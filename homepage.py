import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import gc

# create text2text pipeline for summarization
def llm_pipeline(text, task="summarize"):
  # Model and tokenizer loading (print statement for debugging)
  print("loading model in homepage")
  checkpoint = "models/LaMini-Flan-T5-783M"
  tokenizer = T5Tokenizer.from_pretrained(checkpoint)
  base_model = T5ForConditionalGeneration.from_pretrained(
      checkpoint, device_map='auto', torch_dtype=torch.float32)
  pipe_sum = pipeline(
      'text2text-generation',
      model=base_model,
      tokenizer=tokenizer,
      max_length=400,
      min_length=50)
  
  prompt = f"{task.capitalize()} : {text}"
  result = pipe_sum(prompt)
  result = result[0]['generated_text']
  gc.collect()
  return result

# Sidebar navigation
st.set_page_config(page_title="Synthosnap", page_icon="✒️")

st.title("Summarization and Text Analysis")
st.write("Welcome to Synthosnap, your tool for summarizing and analyzing text.")

# User Input Section
st.header("Input Text")
user_input = st.text_area("Enter the text you want to analyze:", height=200)

# Calculate word count
word_count = len(user_input.split()) if user_input else 0  # Handle empty input

# Display word count
st.write(f"Word count: {word_count}")

# Button to initiate summarization
if st.button("Summarize"):
  if user_input:
    with st.spinner("Summarizing..."):
      summary = llm_pipeline(user_input, task="summarize")
      st.subheader("Summary:")
      placeholder = st.empty()
      placeholder.markdown(summary)
  else:
    st.warning("Please enter some text for analysis.")

# Button to correct grammatical mistakes
if st.button("Correct Grammar"):
  if user_input:
    with st.spinner("Correcting grammar..."):
      corrected_text = llm_pipeline(user_input, task="correct grammar")
      st.subheader("Corrected Text:")
      placeholder = st.empty()
      placeholder.markdown(corrected_text)
  else:
    st.warning("Please enter some text for analysis.")
