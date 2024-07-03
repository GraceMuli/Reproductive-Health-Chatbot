import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv('processed_dataset.csv')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    tokens = text.split()  # Tokenize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Apply text preprocessing
df['cleaned_question'] = df['question'].apply(preprocess_text)

# Perform TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['cleaned_question'].values.astype('U'))  # Assuming 'question' column contains text data
y = df['intent']

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the classifier and set parameters for GridSearchCV
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Evaluate the model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Print classification report
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# Save the TF-IDF vectorizer and trained model
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(best_rf, 'random_forest_model.joblib')
