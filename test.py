import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Function to load TF-IDF vectorizer and trained model
def load_model_and_tfidf(vectorizer_file, model_file):
    tfidf_vectorizer = joblib.load(vectorizer_file)
    model = joblib.load(model_file)
    return tfidf_vectorizer, model

# Function to evaluate the model on test data
def evaluate_model(test_df, tfidf_vectorizer, model):
    X_test = tfidf_vectorizer.transform(test_df['question'].values.astype('U'))
    y_test = test_df['intent']
    
    y_pred = model.predict(X_test)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

# Example usage
if __name__ == "__main__":
    # Paths to saved files
    vectorizer_file = 'tfidf_vectorizer.joblib'
    model_file = 'random_forest_model.joblib'
    
    try:
        # Load TF-IDF vectorizer and model
        tfidf_vectorizer, model = load_model_and_tfidf(vectorizer_file, model_file)
    except FileNotFoundError:
        print(f"Error: Model file '{model_file}' not found. Please check the file path.")
        exit(1)
    
    # Load your test dataset
    test_df = pd.read_csv('test_reproductive_dataset.csv')
    
    # Evaluate the model on the test set
    evaluate_model(test_df, tfidf_vectorizer, model)
