from flask import Flask, render_template, url_for, request
import nltk
from nltk.corpus import brown
from nltk.corpus import PlaintextCorpusReader

nltk.download('brown')
nltk.download('punkt')

app = Flask(__name__)

# Utility functions for language model

def calculate_trigram_frequencies(tokens):
    
    trigram_list = list(nltk.trigrams(tokens))
    bigram_list, next_word_list = zip(*[((a, b), c) for a, b, c in trigram_list]) if trigram_list else ([], [])
    return nltk.ConditionalFreqDist(list(zip(bigram_list, next_word_list)))

def calculate_bigram_frequencies(tokens):
    
    bigram_list = list(nltk.bigrams(tokens))
    return nltk.ConditionalFreqDist(bigram_list)

def add_prediction_if_unique(predictions, candidate):
    
    if all(pred[0] != candidate[0] for pred in predictions):
        predictions.append(candidate)

def handle_partial_input(words, position):
    
    possible_predictions = bigram_frequencies[(words[position - 2])].most_common() if position > 1 else []
    unique_predictions = []
    count = 0
    
    # Direct matches

    for prediction in possible_predictions:
        if prediction[0].startswith(words[position - 1]):
            add_prediction_if_unique(unique_predictions, prediction)
            count += 1
        if count == 3:
            return unique_predictions
    
    # Use edit distance if fewer than 3 predictions

    if len(unique_predictions) < 3:
        potential_matches = [
            (word, nltk.edit_distance(word, words[position - 1], transpositions=True))
            for word, _ in possible_predictions
        ]
        potential_matches.sort(key=lambda x: x[1])
        
        for match in potential_matches:
            if len(unique_predictions) < 3 and match[1] > 0:
                add_prediction_if_unique(unique_predictions, match)
    
    return unique_predictions

# Load corpus and compute n-gram frequencies

corpus_reader = PlaintextCorpusReader('./', '.*')
combined_tokens = brown.words() + corpus_reader.words('chat.txt')

bigram_frequencies = calculate_bigram_frequencies(combined_tokens)
trigram_frequencies = calculate_trigram_frequencies(combined_tokens)

# Flask routes

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('text', "").strip()
    predictions = []
    
    if user_input:
        words = user_input.split()
        num_words = len(words)

        if num_words == 1:
            
            # Single-word input

            predictions = [
                word for word, _ in bigram_frequencies[(words[0])].most_common(5)
            ] if words[0] in bigram_frequencies else []
        elif num_words > 1:

            # Multi-word input

            predictions = [
                word for word, _ in trigram_frequencies[(words[-2], words[-1])].most_common(5)
            ] if (words[-2], words[-1]) in trigram_frequencies else []

        if not predictions:
            predictions = [f"No prediction found for '{user_input}'"]

    return render_template("index.html", s=user_input, t=predictions)

if __name__ == '__main__':
    app.run(debug=True)
