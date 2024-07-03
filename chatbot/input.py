import pandas as pd
import joblib
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker  # Import SpellChecker

# Initialize SpellChecker and WordNet lemmatizer
spell_checker = SpellChecker()
lemmatizer = WordNetLemmatizer()

# Function to load TF-IDF vectorizer and trained model
def load_model_and_tfidf(vectorizer_file, model_file):
    tfidf_vectorizer = joblib.load(vectorizer_file)
    model = joblib.load(model_file)
    return tfidf_vectorizer, model

# Function to lemmatize tokens
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to generate synonyms using WordNet
def generate_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Function to preprocess text: tokenize, lemmatize, and join
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = lemmatize_tokens(tokens)
    return ' '.join(lemmatized_tokens)

# Function to perform spell checking and correction
def correct_spelling(input_text):
    corrected_text = []
    for word in word_tokenize(input_text):
        corrected_word = spell_checker.correction(word)
        if corrected_word.lower() != word.lower():
            print(f"Spell check: Replacing '{word}' with '{corrected_word}'")
        corrected_text.append(corrected_word)
    return ' '.join(corrected_text)

# Function to predict intent and fetch answer from dataset
def predict_intent(question, tfidf_vectorizer, model, df):
    # Correct spelling errors in the question
    corrected_question = correct_spelling(question)
    
    # Preprocess the corrected question
    preprocessed_question = preprocess_text(corrected_question)
    question_tfidf = tfidf_vectorizer.transform([preprocessed_question])
    predicted_intent = model.predict(question_tfidf)[0]
    
    # Fetch all answers based on the predicted intent from the dataset
    possible_answers = df.loc[df['intent'] == predicted_intent, 'answer'].values
    
    # Select best answer based on TF-IDF similarity
    best_answer = select_best_answer(question_tfidf, possible_answers, tfidf_vectorizer)
    
    if best_answer is None:
        return predicted_intent, "I'm sorry, I don't have an answer for that question."
    else:
        return predicted_intent, best_answer

# Function to select the best answer based on TF-IDF similarity
def select_best_answer(question_tfidf, possible_answers, tfidf_vectorizer):
    max_sim = -1
    best_answer = None
    question_tfidf_dense = question_tfidf.toarray()  # Convert sparse matrix to dense array
    
    for answer in possible_answers:
        answer_tfidf = tfidf_vectorizer.transform([answer])
        answer_tfidf_dense = answer_tfidf.toarray()  # Convert sparse matrix to dense array
        
        # Compute cosine similarity (taking the first element since it's a single comparison)
        similarity = cosine_similarity(question_tfidf_dense, answer_tfidf_dense)[0][0]
        
        if similarity > max_sim:
            max_sim = similarity
            best_answer = answer
    
    return best_answer

# Main function to handle user interaction
def main():
    # Paths to saved files
    vectorizer_file = 'tfidf_vectorizer.joblib'
    model_file = 'random_forest_model.joblib'
    
    try:
        # Load TF-IDF vectorizer and model
        tfidf_vectorizer, model = load_model_and_tfidf(vectorizer_file, model_file)
    except FileNotFoundError:
        print(f"Error: Model file '{model_file}' not found. Please check the file path.")
        return
    
    # Load your processed dataset or test dataset
    dataset_file = 'processed_dataset.csv'
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_file}' not found. Please check the file path.")
        return
    
    print("Chatbot is ready! You can start asking questions. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input == 'exit':
            print("Exiting...")
            break
        
        # Predict intent and fetch answer
        predicted_intent, answer = predict_intent(user_input, tfidf_vectorizer, model, df)
        
        # Print bot response
        print(f"Bot (Intent: {predicted_intent}): {answer}")

if __name__ == "__main__":
    main()
