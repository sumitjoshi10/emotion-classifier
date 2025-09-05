import streamlit as st
import pandas as pd
import numpy as np

from transformers import BertForSequenceClassification, BertTokenizer

st.set_page_config(page_title="Emotion Classifier", page_icon="😃")

st.title("BERT Emotion Classifier")


save_directory = './artifacts'


model_saved = BertForSequenceClassification.from_pretrained(save_directory)
tokenizer_saved = BertTokenizer.from_pretrained(save_directory)


def predict_emotion(text):
    inputs = tokenizer_saved(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model_saved(**inputs)
    pred_id = outputs.logits.argmax(dim=-1).item()
    return model_saved.config.id2label[pred_id]

text = st.text_area("Enter your Text Here")

if st.button("Predict"):
    result = predict_emotion(text)
    st.write(result)
