import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## Loading the Tensorflow model for prediction

model = load_model('model.h5')

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

st.title("Tweets Sentiment Analysis")

tweet = st.text_area('Enter The Tweet:')

if st.button('Predict Sentiment') and tweet.strip():
    sequence = tokenizer.texts_to_sequences([tweet])
    sequences = pad_sequences(sequence, maxlen=99,padding='post')

    prediction = model.predict(sequences)
    predicted_class = np.argmax(prediction, axis=1)[0]

    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    st.write(f'Sentiment: {sentiment_map[predicted_class]}')  