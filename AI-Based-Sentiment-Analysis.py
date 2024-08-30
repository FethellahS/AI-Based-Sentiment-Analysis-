import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Load pre-trained LSTM model
model = load_model('path_to_your_lstm_model.h5')

# Define tokenizer and maximum length (these should match those used during training)
tokenizer = Tokenizer(num_words=5000)  # Adjust num_words based on your model
MAX_LENGTH = 100  # Adjust MAX_LENGTH based on your model

def preprocess_text(text):
    # Clean and preprocess text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\d', '', text)  # Remove numbers
    text = text.strip()  # Remove leading and trailing spaces
    return text

def predict_sentiment(texts):
    processed_texts = [preprocess_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH)

    predictions = model.predict(padded_sequences)
    sentiment_labels = ['positive', 'neutral', 'negative']
    sentiment_results = [sentiment_labels[np.argmax(pred)] for pred in predictions]

    return sentiment_results

def main():
    # Example social media posts
    posts = [
        "I love this new phone! It's amazing.",
        "I'm not sure how I feel about the latest update.",
        "I hate waiting for my package to arrive."
    ]

    sentiments = predict_sentiment(posts)

    for post, sentiment in zip(posts, sentiments):
        print(f"Post: {post}\nSentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
