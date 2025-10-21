from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tweet = request.form['tweet']
        if not tweet.strip():
            return render_template('index.html', error="Please enter a tweet before analyzing.")

        # Basic fake sentiment logic (replace later with ML model)
        tweet_lower = tweet.lower()
        if any(word in tweet_lower for word in ['love', 'good', 'happy', 'great', 'awesome']):
            sentiment = "Positive 😀"
        elif any(word in tweet_lower for word in ['bad', 'hate', 'sad', 'terrible', 'awful']):
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        return render_template('index.html', tweet=tweet, prediction=sentiment)

    except Exception as e:
        return render_template('index.html', error=f"Something went wrong: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
