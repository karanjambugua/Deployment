from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import chardet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') 
nltk.download('vader_lexicon')
app = Flask(__name__)


# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
# Load the stopwords for text cleaning
stop_words = set(stopwords.words('english'))
# Handle negations in preprocessing
def handle_negations(text):
    # Replace negation words with '_not_' to better capture the sentiment
    negations = ["don't", "doesn't", "can't", "isn't", "aren't", "won't", "didn't", "hasn't", "haven't"]
    for neg in negations:
        if neg in text.lower():
            text = text.replace(neg, "not_{}".format(neg))  # Replace with a special marker
    return text

# Define function to preprocess tweet text
def preprocess_text(text):
    # Ensure the input is a string, and if not, return an empty string
    if not isinstance(text, str):
        return ''
    
    # Handle negations first (before other preprocessing steps)
    text = handle_negations(text)

    # Basic text preprocessing: remove non-alphabetic characters and stopwords
    text = ' '.join([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])
    
    return text


# Define and train Logistic Regression model
def train_logistic_regression_model():
    # Read the dataset
    df = pd.read_csv('Data/judge-1377884607_tweet_product_company.csv', encoding='Latin-1')
    
    # Preprocess tweet texts
    df['tweet_text'] = df['tweet_text'].apply(preprocess_text)

    # Define features (X) and labels (y)
    X = df['tweet_text']  # Features (tweet texts)
    y = df['is_there_an_emotion_directed_at_a_brand_or_product']  # Labels (sentiment)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a pipeline for vectorization + model training
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(pipeline, 'sentiment_model.pkl')

# Call the function to train the model (only once)
train_logistic_regression_model()

# Load the trained model when the app starts
model = joblib.load('sentiment_model.pkl')
# Load the Logistic Regression model (only once during app initialization)
def load_logistic_regression_model():
    # Load your pre-trained Logistic Regression model here
    model = joblib.load('sentiment_model.pkl')
    return model

logistic_model = load_logistic_regression_model()

# Define function for Logistic Regression prediction
def logistic_regression_predict(text):
    preprocessed_text = preprocess_text(text)
    sentiment = logistic_model.predict([preprocessed_text])[0]
    return sentiment


# Home route: Renders the index.html page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')
# Sentiment analysis route: Accepts tweet input and returns sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()  # Get JSON data from the frontend
    tweet = data.get("tweet")  # Extract tweet text
    
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Preprocess tweet text before analysis
    preprocessed_tweet = preprocess_text(tweet)

    # Analyze the sentiment using VADER
    sentiment_score = analyzer.polarity_scores(preprocessed_tweet)
    compound_score = sentiment_score['compound']  # Get the compound score (ranges from -1 to 1)

    # Determine sentiment based on the compound score
    if compound_score >= 0.05:
        sentiment = 'Positive emotion'
    elif compound_score <= -0.05:
        sentiment = 'Negative emotion'
    else:
        sentiment = 'Neutral emotion'

    # Return sentiment result
    return jsonify({"sentiment": sentiment, "score": sentiment_score})

# Detect file encoding (optional, for debugging purposes)
with open('Data/judge-1377884607_tweet_product_company.csv', 'rb') as f:
    result = chardet.detect(f.read())
print(result)

if __name__ == '__main__':
    app.run(debug=True)