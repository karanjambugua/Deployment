import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import re
# Data Preprocessing (simplified)
def preprocess_data():
    # Load your data
    df = pd.read_csv('Data\\judge-1377884607_tweet_product_company.csv', encoding="Latin-1")
    df.dropna(subset=["tweet_text"], inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Cleaning tweets (simplified)
    df['clean_text'] = df['tweet_text'].apply(lambda x: re.sub(r"[^A-Za-z\s]", "", x))

    return df

# Feature and Target
df = preprocess_data()
X = df['clean_text']
y = df['is_there_an_emotion_directed_at_a_brand_or_product']

# Label Encoding for y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Logistic Regression Model
model = Pipeline([
    ('tfidf', tfidf),
    ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))
])

# Train the model
model.fit(X_train, y_train)

# Save the model for later use (in Flask)
joblib.dump(model, 'sentiment_model.pkl')

# Example for prediction
def predict_sentiment(tweet):
    model = joblib.load('sentiment_model.pkl')
    return label_encoder.inverse_transform(model.predict([tweet]))[0]
