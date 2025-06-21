import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import gensim
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import warnings
from nltk.corpus import stopwords
import string
from a_detection.utils import preprocess_text

# Explanation: Download NLTK resources if not already present.
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource.startswith('punkt') else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Load datasets
fake = pd.read_csv('fake.csv')
true = pd.read_csv('true.csv')

# Add labels: 0 for fake, 1 for true
fake['label'] = 0
true['label'] = 1

# Combine datasets
all_news = pd.concat([fake, true], ignore_index=True)

# Explanation: Preprocess text data by filling missing values and combining title and text into a single 'content' column.
all_news['title'] = all_news['title'].fillna('')
all_news['text'] = all_news['text'].fillna('')
all_news['content'] = all_news['title'] + ' ' + all_news['text']

# Features and labels
X = all_news['content']
y = all_news['label']

# Explanation: Split the dataset into training and testing sets, including the full 'all_news' dataframe to access title and text for similarity calculation.
X_train, X_test, y_train, y_test, news_train, news_test = train_test_split(X, y, all_news, test_size=0.2, random_state=42, stratify=y)

# --- Text Preprocessing ---
# The preprocess_text function is now imported from a_detection.utils

# --- TF-IDF Vectorization & Explainer Model ---
print("Fitting TF-IDF vectorizer and training explainer model...")
# The vectorizer's tokenizer will use our custom preprocessing function.
# We join the tokens back into a string as TfidfVectorizer works on raw text.
vectorizer = TfidfVectorizer(max_features=10000, tokenizer=preprocess_text)

# Fit on training data and transform both sets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train and save the TF-IDF based explainer model
explainer_model = LogisticRegression(max_iter=1000)
explainer_model.fit(X_train_tfidf, y_train)

joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(explainer_model, 'explainer_model.joblib')
print("TF-IDF vectorizer and explainer model saved.")


# --- Word2Vec Model & Main Prediction Model ---
print("\nTraining Word2Vec model...")
# We can reuse the preprocessing from the TF-IDF step's tokenizer call, but for clarity we'll show it here.
tokenized_train_data = [preprocess_text(doc) for doc in X_train]
word2vec_model = gensim.models.Word2Vec(tokenized_train_data, vector_size=100, window=5, min_count=2, workers=4)
word2vec_model.save("word2vec.model")
print("Word2Vec model saved.")

# Explanation: This function converts a document into a single vector by averaging the Word2Vec vectors of its words.
def document_vector(doc, model):
    # Preprocess the document and remove out-of-vocabulary words
    processed_tokens = preprocess_text(doc)
    doc_vectors = [model.wv[word] for word in processed_tokens if word in model.wv.key_to_index]
    if not doc_vectors:
        return np.zeros(model.vector_size)
    return np.mean(doc_vectors, axis=0)

# Explanation: Create document vectors for the training and test sets.
X_train_vec = np.array([document_vector(doc, word2vec_model) for doc in X_train])
X_test_vec = np.array([document_vector(doc, word2vec_model) for doc in X_test])

# --- Cosine Similarity (Demonstration using Word2Vec) ---
print("\nDemonstrating Cosine Similarity between title and text using Word2Vec:")
# Explanation: This section demonstrates calculating semantic similarity on a few test samples using Word2Vec.
sample_test_news = news_test.head(3)
for index, row in sample_test_news.iterrows():
    title = row['title']
    text = row['text']
    
    if title and text:
        # Vectorize title and text using the trained Word2Vec model
        title_vec = document_vector(title, word2vec_model).reshape(1, -1)
        text_vec = document_vector(text, word2vec_model).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(title_vec, text_vec)[0][0]
        
        print(f"\nTitle: '{title[:50]}...'")
        print(f"Semantic Similarity with its text: {similarity_score:.2f} ({similarity_score*100:.0f}% confident match)")
    else:
        print(f"\nSkipping a row due to empty title or text.")

# --- Main Classification Model Training (Using Word2Vec) ---
print("\nTraining Main Logistic Regression model with Word2Vec features...")
main_model = LogisticRegression(max_iter=1000)
main_model.fit(X_train_vec, y_train)

# Evaluate Main Model
print("\nMain Model (Word2Vec) Evaluation:")
y_pred_main = main_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_main))

# Save the main classification model.
joblib.dump(main_model, 'model.joblib')
print('Main classification model saved as model.joblib')

print("\n--- Training complete ---") 