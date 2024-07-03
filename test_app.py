import pandas as pd
import numpy as np
import spacy
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask application
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('processed_dataset.csv')  # Assuming your dataset is loaded correctly

def preprocess_question(question):
    """Preprocesses the question to ensure consistency with training data."""
    doc = nlp(question)
    return doc.vector

def classify_intent(question):
    """Classifies the intent of the question and retrieves the corresponding answer."""
    question_vec = preprocess_question(question)
    
    # Calculate cosine similarity with all question vectors in the dataset
    similarities = []
    for _, row in df.iterrows():
        sim = cosine_similarity([question_vec.reshape(1, -1)], [nlp(row['question']).vector.reshape(1, -1)])[0][0]
        similarities.append(sim)
    
    most_similar_index = np.argmax(similarities)
    predicted_intent = df.iloc[most_similar_index]['intent']
    answer = df.iloc[most_similar_index]['answer']
    
    return predicted_intent, answer

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        data = request.get_json()
        question = data['question']
        
        predicted_intent, answer = classify_intent(question)
        
        response = {
            'question': question,
            'intent': predicted_intent,
            'answer': answer
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
