from nltk.tokenize import word_tokenize
import wikipedia
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Fetch 5 Wikipedia documents
topics = [
    "Decision problem",
    "Algorithm",
    "Data structure",
    "Complexity theory",
    "Graph theory"
]

documents = []
for topic in topics:
    try:
        page = wikipedia.page(topic).content
        text = "\n".join(page.splitlines()[:5])[:1000]  # Limit to 1000 characters
        tokens = word_tokenize(text.lower())  # Tokenize and normalize
        cleaned_tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
        documents.append(cleaned_tokens)  # Store the cleaned text
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for {topic}: {e.options[0]}")
    except Exception as e:
        print(f"Error fetching page for {topic}: {e}")

# Train Word2Vec model on the tokenized documents
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# Function to compute average word vectors for each document
def get_avg_word2vec(doc, model):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Create feature vectors for each document
document_vectors = [get_avg_word2vec(doc, model) for doc in documents]

# Prepare labels (just for demonstration, assuming topic index as label)
labels = [0, 1, 2, 3, 4]  # Labels for each document (0 for 'Decision problem', 1 for 'Algorithm', etc.)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(document_vectors, labels, test_size=0.2, random_state=42)

# Train Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

# You can also display the predicted vs true labels for further analysis
# predicted_df = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})
# print(predicted_df)

# Check the size of train and test sets
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Check a sample document vector to see if it's reasonable
print("Sample document vector (first document):")
print(document_vectors[0])

# Print the true and predicted labels to ensure they match the intended output
predicted_df = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})
print(predicted_df)

# Example for cross-validation (with more data in your actual case)
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, document_vectors, labels, cv=5)  # 5-fold cross-validation

print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean()}")