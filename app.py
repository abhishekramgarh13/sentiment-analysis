from flask import Flask, request, jsonify, render_template
import os
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained vectorizer and model
MODELS_DIR = os.path.join(os.getcwd(), 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')

# Load the vectorizer and model
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_comment(comment):
    # Convert to lowercase
    comment = comment.lower()

    # Remove trailing and leading whitespaces
    comment = comment.strip()

    # Remove URLs
    comment = re.sub(r'https?://\S+|www\.\S+', '', comment)

    # Remove HTML tags
    comment = re.sub(r'<.*?>', '', comment)

    # Remove newline characters
    comment = re.sub(r'\n', ' ', comment)

    # Remove non-alphanumeric characters and punctuation
    comment = re.sub(r'[^\w\s]', '', comment)

    # Remove stopwords but retain important ones for sentiment analysis
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

# Route to serve the UI
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the request
    data = request.json
    review = data.get('review', '')

    # Preprocess the review
    processed_review = preprocess_comment(review)

    # Vectorize the review
    review_vectorized = vectorizer.transform([processed_review])

    # Make prediction
    prediction = model.predict(review_vectorized)

    # Map prediction to sentiment
    sentiment = 'positive' if prediction[0] == 1 else 'negative'

    # Return the result as JSON
    return jsonify({'sentiment': sentiment})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
