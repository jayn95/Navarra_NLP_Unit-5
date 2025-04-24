import math
import pandas as pd
from raw_frequency import fetch_documents, compute_tf, create_vocab
from tf_idf import compute_idf, compute_tfidf

# Topics and fetching documents
topics = ["Decision problem", "Algorithm", "Data structure", "Complexity theory", "Graph theory"]
documents = fetch_documents(topics)

# Build vocabulary and compute TF-IDF
vocab = create_vocab(documents)
tf_matrix = [compute_tf(doc, vocab) for doc in documents]
idf = compute_idf(documents, vocab)
tfidf_matrix = [compute_tfidf(tf, idf, vocab) for tf in tf_matrix]

# Cosine Similarity function
def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1_len = math.sqrt(sum(vec1[term] ** 2 for term in vocab))
    vec2_len = math.sqrt(sum(vec2[term] ** 2 for term in vocab))
    if vec1_len == 0 or vec2_len == 0:
        return 0.0
    return dot_product / (vec1_len * vec2_len)

# Create and fill similarity matrix
similarity_matrix = pd.DataFrame(index=topics, columns=topics, dtype=float)

for i in range(len(documents)):
    for j in range(len(documents)):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j], vocab)
        similarity_matrix.iloc[i, j] = round(sim, 4)

# Display result
print("Cosine Similarity Matrix:")
print(similarity_matrix.to_string())
