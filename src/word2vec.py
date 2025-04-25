# Uses the same dataset used for raw and tf_idf, 
# uses the word2vec package to create a classifier for dense vectors.
# Uses Logistic Regression, with the appropriate configuration for the model and dataset.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Sample dataset of topics and their descriptions
topics = [
    ("Decision problem", "A decision problem is a problem that can be posed as a yes/no question."),
    ("Algorithm", "An algorithm is a step-by-step procedure for solving a problem."),
    ("Data structure", "A data structure is a way to organize and store data for efficient access and modification."),
    ("Complexity theory", "Complexity theory studies the complexity of computational problems."),
    ("Graph theory", "Graph theory is the study of graphs, which are mathematical structures used to model pairwise relations between objects.")
]

# Prepare the dataset
data = pd.DataFrame(topics, columns=["label", "text"])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Tokenize the text data
data['tokens'] = data['text'].apply(preprocess_text)

# Train the Word2Vec model on the tokens
word2vec_model = Word2Vec(sentences=data['tokens'], vector_size=100, window=5, min_count=1, sg=0)

# Function to get the average vector for a sentence
def get_sentence_vector(tokens, model):
    vec = np.zeros(100)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count > 0:
        vec /= count
    return vec

# Create the feature vectors for the dataset
data['vector'] = data['tokens'].apply(lambda x: get_sentence_vector(x, word2vec_model))

# Prepare the features and labels for training
X = np.vstack(data['vector'].values)  # Convert list of vectors into a numpy array
y = data['label'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display all predictions
for true_label, predicted_label in zip(y_test, y_pred):
    print(f"True Label: {true_label} | Predicted Label: {predicted_label}")
