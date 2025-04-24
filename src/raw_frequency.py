# 1. Uses Wikipedia as the corpus, obtains 5 different topics that serve as the documents, 
# and creates a term-document matrix.
# Term-document matrix using raw frequency. 


from collections import Counter
from nltk.tokenize import word_tokenize
import wikipedia
import pandas as pd

# Fetch 5 Wikipedia page
topics = ["Decision problem", "Algorithm", "Data structure", "Complexity theory", "Graph theory"]

def fetch_documents(topics):
    documents = []
    for topic in topics:
        try:
            page = wikipedia.page(topic).content
            text = "\n".join(page.splitlines()[:5])[:1000]  # Limit to 1000 characters
            tokenized_text = word_tokenize(text.lower())  # Tokenize and normalize
            cleaned_text = [word for word in tokenized_text if word.isalnum()]  # Remove punctuation
            documents.append(cleaned_text)  # Store the cleaned text
            # print(f"Extracted text for {topic}:\n{text}\n")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for {topic}: {e.options[0]}")
        except Exception as e:
            print(f"Error fetching page for {topic}: {e}")
    return documents

documents = fetch_documents(topics)
# Vocabulary
vocab = sorted(set(word for doc in documents for word in doc))  # Unique words across all documents

# Compute raw term frequency
def compute_tf(tokens, vocab):
    count = Counter(tokens)
    return {term: count[term] for term in vocab}

# Raw Term-Document Matrix
tf_matrix = [compute_tf(doc, vocab) for doc in documents]  # just one document

# Print the term-frequency vector for one document
# print("Term Frequency (Raw Frequenct):\n")
# for i, ftf_vector in enumerate(tf_matrix):
#     print(f"Document {i+1}: {topics[i]}")
#     for term, freq in ftf_vector.items():
#         if freq > 0:
#             print(f"  {term:15} : {freq}")
#         print("-" * 40)

df = pd.DataFrame(tf_matrix, index=[f'Doc{i+1}' for i in range(len(tf_matrix))])
df = df.fillna(0).astype(int)
print(df)

def create_vocab(documents):
    return sorted(set(word for doc in documents for word in doc))  # Unique words across all documents
