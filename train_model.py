# sentiment_analysis.py
# ----------------------
# Sentiment Analysis on Tweets (Sample Project)
#
# This script performs basic sentiment analysis on a small set of tweets.
# It uses Logistic Regression for classification and TF-IDF for text representation.

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download stopwords if not available
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# 1️⃣ SAMPLE DATASET
# -----------------------------
data = {
    'tweet': [
        "I love this phone! It's awesome ❤️",
        "Worst customer service ever!",
        "The movie was okay, not great but not bad either.",
        "I'm so happy with my new laptop 😍",
        "I hate waiting in long lines 😡",
        "Such a fantastic experience at the restaurant!",
        "The weather is terrible today.",
        "Nothing special, just another day.",
        "Absolutely loved the concert last night!",
        "My internet keeps disconnecting 😤"
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative'
    ]
}

df = pd.DataFrame(data)
print("✅ Sample Data Loaded:")
print(df.head(), "\n")

# -----------------------------
# 2️⃣ TEXT CLEANING FUNCTION
# -----------------------------
def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)            # remove URLs
    text = re.sub(r'@\w+', '', text)               # remove mentions
    text = re.sub(r'#', '', text)                  # remove hashtags symbol
    text = re.sub(r'[^A-Za-z\s]', '', text)        # remove punctuation & numbers
    text = text.lower()                            # lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['clean_tweet'] = df['tweet'].apply(clean_tweet)

# -----------------------------
# 3️⃣ SPLIT DATA
# -----------------------------
X = df['clean_tweet']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 4️⃣ TEXT VECTORIZATION (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5️⃣ TRAIN THE MODEL
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -----------------------------
# 6️⃣ EVALUATE THE MODEL
# -----------------------------
y_pred = model.predict(X_test_vec)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 7️⃣ TEST CUSTOM INPUTS
# -----------------------------
sample_tweets = [
    "I am so excited for the weekend!",
    "This app is terrible, it keeps crashing.",
    "It's just an average day."
]

sample_clean = [clean_tweet(t) for t in sample_tweets]
sample_vec = vectorizer.transform(sample_clean)
predictions = model.predict(sample_vec)

print("\n💬 Sentiment Predictions:")
for tweet, pred in zip(sample_tweets, predictions):
    print(f"Tweet: {tweet}\nPredicted Sentiment: {pred}\n")
