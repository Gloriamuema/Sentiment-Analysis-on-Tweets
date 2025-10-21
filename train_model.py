import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# Sample dataset
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

def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['clean_tweet'] = df['tweet'].apply(clean_tweet)

X = df['clean_tweet']
y = df['sentiment']

vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully!")
