# Uses Wikipedia as the corpus, obtains 5 different topics that serves as the documents, 
# and creates a term-document matrix. 
# Term-document matrix using TF-IDF weights.


from collections import Counter
from nltk.tokenize import word_tokenize
import wikipedia
from math import log
import pandas as pd

# This tf_idf only outputs binary which is 0 and 1

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

# Build vocabulary
vocab = sorted(set(word for doc in documents for word in doc))

# SCompute IDF
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))  # Add smoothing to avoid division by zero
    return idf_dict

idf = compute_idf(documents, vocab)

# Compute normalized TF (term frequency)
def compute_tf(doc, vocab):
    tf_raw = Counter(doc)
    total_terms = len(doc)
    return {term: tf_raw[term] / total_terms for term in vocab}

# Compute TF-IDF
def compute_tfidf(tf, idf, vocab):
    return {term: tf[term] * idf[term] for term in vocab}

tf_matrix = [compute_tf(doc, vocab) for doc in documents]
tfidf_matrix = [compute_tfidf(tf, idf, vocab) for tf in tf_matrix]

# Display output
tfidf_df = pd.DataFrame(tfidf_matrix, columns=vocab, index=topics)
print("TF-IDF Matrix (with normalized TF):")
print(tfidf_df)
print("-" * 40)
