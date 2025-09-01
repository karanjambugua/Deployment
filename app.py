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

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the stopwords for text cleaning
stop_words = set(stopwords.words('english'))

# Define function to preprocess tweet text
def preprocess_text(text):
    # Ensure the input is a string, and if not, return an empty string
    if not isinstance(text, str):
        return ''
    
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
    data = request.get_json()
    tweet = data.get("tweet")
    
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Preprocess and predict sentiment using the Logistic Regression model
    preprocessed_tweet = preprocess_text(tweet)
    sentiment = model.predict([preprocessed_tweet])[0]

    return jsonify({"sentiment": sentiment})

# Detect file encoding (optional, for debugging purposes)
with open('Data/judge-1377884607_tweet_product_company.csv', 'rb') as f:
    result = chardet.detect(f.read())
print(result)

if __name__ == '__main__':
    app.run(debug=True)
