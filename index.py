import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from flask import Flask, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from flask_cors import CORS
nltk.download('punkt')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load the dataset of questions and answers
df = pd.read_csv('data.csv')

# Combine the questions and answers into a single corpus
corpus = df['questions'].tolist() + df['answers'].tolist()

# Preprocess the corpus
corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Vectorize the corpus using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
X = vectorizer.fit_transform(corpus)

# Set the similarity threshold
SIMILARITY_THRESHOLD = 0.6


@app.route('/chat/<input>', methods=['GET'])
def chatbot(input):
    # Preprocess the user input
    input_tokens = word_tokenize(input.lower())

    # Vectorize the user input using the same vectorizer
    input_vector = vectorizer.transform([input_tokens])

    # Calculate cosine similarity between the user input and the corpus
    similarities = cosine_similarity(input_vector, X)

    # Find the most similar sentence in the corpus
    most_similar_index = np.argmax(similarities)
    most_similar_sentence = corpus[most_similar_index]

    # Calculate the similarity between the user input and the most similar sentence
    similarity = similarities[0, most_similar_index]

    # Find the corresponding question or answer
    if most_similar_index < len(df):
        # The most similar sentence is a question
        response = df.loc[most_similar_index, 'answers']
    else:
        # The most similar sentence is an answer
        response = df.loc[most_similar_index - len(df), 'questions']

    # If the similarity is below the threshold, we don't understand
    if similarity < SIMILARITY_THRESHOLD:
        response = "Sorry, I don't understand. Please try writing something else."

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run()
