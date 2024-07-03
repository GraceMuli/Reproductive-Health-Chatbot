from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)
CORS(app)

# Load the dataset
df = pd.read_csv('processed_dataset.csv')

# Text preprocessing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation and token not in stop_words]
    return ' '.join(tokens)

# Encode intents
le = LabelEncoder()
df['encoded_intent'] = le.fit_transform(df['intent'])

# Load or train the DistilBERT model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(df['encoded_intent'].unique()))

# Function to tokenize and preprocess text
def tokenize_and_preprocess(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='tf'
    )

# Function to predict intent
def predict_intent(question):
    processed_question = preprocess_text(question)
    inputs = tokenize_and_preprocess(processed_question)
    outputs = model(inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_intent_index = tf.argmax(probabilities, axis=-1).numpy()[0]
    confidence = probabilities[0][predicted_intent_index].numpy()
    predicted_intent = le.inverse_transform([predicted_intent_index])[0]
    return predicted_intent, float(confidence)

# Function to find best answer
def find_best_answer(question, intent):
    relevant_answers = df[df['intent'] == intent]['answer'].tolist()
    if not relevant_answers:
        return "I'm sorry, I don't have an answer for that intent."
    
    question_words = set(preprocess_text(question).split())
    best_answer = max(relevant_answers, key=lambda x: len(set(preprocess_text(x).split()) & question_words))
    return best_answer

# Endpoint to handle incoming queries
@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Predict intent
    predicted_intent, confidence = predict_intent(question)
    
    # Find best answer
    answer = find_best_answer(question, predicted_intent)

    return jsonify({
        'question': question,
        'intent': predicted_intent,
        'answer': answer,
        'confidence': confidence
    })

if __name__ == '__main__':
    # Training or loading model code would typically go here
    
    # Save the model after training or loading
    model.save_pretrained('./distilbert_intent_model')
    
    # Run the Flask app
    app.run(debug=True)
