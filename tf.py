import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# File paths
MODEL_PATH_PREFIX = 'intent_classification_model_fold_'
TOKENIZER_PATH = 'tokenizer.pickle'
LABEL_ENCODER_PATH = 'label_encoder.pickle'

# Text preprocessing using spaCy
nlp = spacy.load('en_core_web_sm')

def preprocess_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(tokens)

# Custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Data Augmentation functions
def augment_text(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    return augmented_text[0] if isinstance(augmented_text, list) else augmented_text

def spell_correct_text(text):
    aug = nac.KeyboardAug()
    corrected_text = aug.augment(text)
    return corrected_text[0] if isinstance(corrected_text, list) else corrected_text

# Define model creation function
def create_model(num_words, embedding_dim, embedding_matrix, num_classes):
    model = Sequential([
        Embedding(num_words, embedding_dim, embeddings_initializer=Constant(embedding_matrix), input_length=50, trainable=True, mask_zero=True),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        Bidirectional(LSTM(16, return_sequences=True)),
        AttentionLayer(),
        Dropout(0.6),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Function to load tokenizer and label encoder
def load_tokenizer_and_label_encoder():
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(LABEL_ENCODER_PATH, 'rb') as handle:
        le = pickle.load(handle)
    return tokenizer, le

# Force retraining (you can make this a command-line argument or environment variable)
force_retrain = True

if force_retrain or not os.path.exists(f'{MODEL_PATH_PREFIX}1.keras'):
    print("Training new models...")
    # Load and preprocess the dataset
    df = pd.read_csv('processed_dataset.csv')

    # Apply data augmentation
    df['augmented_question'] = df['question'].apply(lambda x: augment_text(x))
    df['corrected_question'] = df['question'].apply(lambda x: spell_correct_text(x))

    # Combine original and augmented data
    df_augmented = pd.concat([df, df[['augmented_question', 'intent']].rename(columns={'augmented_question': 'question'})])
    df_augmented = pd.concat([df_augmented, df[['corrected_question', 'intent']].rename(columns={'corrected_question': 'question'})])

    # Encode intents
    le = LabelEncoder()
    df_augmented['intent_encoded'] = le.fit_transform(df_augmented['intent'])

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df_augmented['question'].apply(preprocess_text_spacy))
    X_seq = tokenizer.texts_to_sequences(df_augmented['question'].apply(preprocess_text_spacy))
    X_padded = pad_sequences(X_seq, maxlen=50)
    y_categorical = to_categorical(df_augmented['intent_encoded'])

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_padded, df_augmented['intent_encoded'])
    y_resampled_categorical = to_categorical(y_resampled)

    # Load pre-trained GloVe embeddings
    embedding_dim = 100
    embedding_index = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    word_index = tokenizer.word_index
    num_words = min(5000, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= 5000:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Implement k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    models = []  # Initialize an empty list to store trained models

    for fold, (train_index, val_index) in enumerate(kfold.split(X_resampled)):
        print(f"Training fold {fold + 1}")
        
        X_train, X_val = X_resampled[train_index], X_resampled[val_index]
        y_train, y_val = y_resampled_categorical[train_index], y_resampled_categorical[val_index]
        
        model = create_model(num_words, embedding_dim, embedding_matrix, len(le.classes_))
        
        # Optimizer and compilation
        optimizer = AdamW(learning_rate=1e-3, weight_decay=0.01, clipnorm=1.0)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
        
        # Callbacks
        checkpoint = ModelCheckpoint(f'{MODEL_PATH_PREFIX}{fold+1}.keras', monitor='val_loss', save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        csv_logger = CSVLogger(f'training_log_fold_{fold+1}.csv')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        
        # Train the model
        history = model.fit(X_train, y_train, 
                            epochs=30, batch_size=32, 
                            validation_data=(X_val, y_val),
                            callbacks=[checkpoint, early_stopping, csv_logger, reduce_lr])
        
        # Save the model
        model.save(f'{MODEL_PATH_PREFIX}{fold+1}.keras')
        models.append(model)

    # Save tokenizer and label encoder
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(LABEL_ENCODER_PATH, 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print("Loading pre-trained models...")
    # Load pre-trained models and associated files
    # Your code to load pre-trained models goes here
    # Load tokenizer and label encoder
    tokenizer, le = load_tokenizer_and_label_encoder()

# Function to predict intent using the ensemble of models
def predict_intent(question):
    processed_question = preprocess_text_spacy(question)
    question_seq = tokenizer.texts_to_sequences([processed_question])
    question_padded = pad_sequences(question_seq, maxlen=50)
    
    predictions = np.zeros((1, len(le.classes_)))
    for model in models:
        predictions += model.predict(question_padded)
    
    predictions /= len(models)
    predicted_intent_index = np.argmax(predictions[0])
    return le.inverse_transform([predicted_intent_index])[0], predictions[0][predicted_intent_index]

# Function to find best answer
def find_best_answer(question, intent):
    df = pd.read_csv('processed_dataset.csv')  # Load this only when needed
    relevant_answers = df[df['intent'] == intent]['answer'].tolist()
    if not relevant_answers:
        return "I'm sorry, I don't have an answer for that intent."

    question_words = set(preprocess_text_spacy(question).split())
    best_answer = max(relevant_answers, key=lambda x: len(set(preprocess_text_spacy(x).split()) & question_words))
    return best_answer

# Endpoint to handle incoming queries
@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Predict intent and find best answer
    predicted_intent, confidence = predict_intent(question)
    answer = find_best_answer(question, predicted_intent)

    # Debug information
    debug_info = {
        'processed_question': preprocess_text_spacy(question),
        'predicted_intent': predicted_intent,
        'confidence': float(confidence),
        'unique_intents': len(le.classes_),
        'total_models': len(models),
        'model_type': 'keras_ensemble'
    }

    return jsonify({
        'question': question,
        'intent': predicted_intent,
        'answer': answer,
        'debug_info': debug_info
    })

if __name__ == '__main__':
    app.run(debug=True)
