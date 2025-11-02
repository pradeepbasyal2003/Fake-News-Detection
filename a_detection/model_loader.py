import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from captum.attr import IntegratedGradients
from nltk.corpus import stopwords
import nltk
import numpy as np
import re
import os

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "gru_classifier.pth")
W2V_PATH = os.path.join(BASE_DIR, "GoogleNews-vectors-negative300.bin.gz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 300
# =================================================


# ============ TOKENIZATION ============
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


# ============ MODEL DEFINITION ============
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ============ LOAD MODEL & W2V ============
print("Loading GRU model and word vectors...")
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
model = GRUClassifier(input_dim=EMBEDDING_DIM, hidden_dim=128,
                      num_layers=1, output_dim=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")


# ============ TEXT TO VECTOR ============
def text_to_vectors(text, keyed_vectors, max_len):
    tokens = tokenize(text)
    vectors = []
    for token in tokens[:max_len]:
        if token in keyed_vectors:
            vec = torch.tensor(keyed_vectors[token][:EMBEDDING_DIM], dtype=torch.float32)
        else:
            vec = torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
        vectors.append(vec)

    while len(vectors) < max_len:
        vectors.append(torch.zeros(EMBEDDING_DIM, dtype=torch.float32))

    return torch.stack(vectors).unsqueeze(0).to(device), tokens[:max_len]


# ============ COSINE SIMILARITY ============
def compute_similarity(title, body):
    title_vec, _ = text_to_vectors(title, w2v, MAX_SEQ_LEN)
    body_vec, _ = text_to_vectors(body, w2v, MAX_SEQ_LEN)
    title_mean = title_vec.mean(dim=1).cpu().numpy()
    body_mean = body_vec.mean(dim=1).cpu().numpy()
    return float(cosine_similarity(title_mean, body_mean)[0][0])


# ============ INTEGRATED GRADIENTS ============
def get_influential_words_IG(model, text, keyed_vectors, max_len, top_k=10):
    tokens = [t for t in text.lower().split() if t.isalpha() and t not in stop_words]

    vectors = np.array([
        keyed_vectors[t] if t in keyed_vectors else np.zeros(keyed_vectors.vector_size)
        for t in tokens
    ])

    if len(vectors) < max_len:
        pad_len = max_len - len(vectors)
        vectors = np.vstack([vectors, np.zeros((pad_len, keyed_vectors.vector_size))])
    else:
        vectors = vectors[:max_len]

    sequence = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).to(device)

    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(
        sequence,
        baselines=torch.zeros_like(sequence),
        return_convergence_delta=True
    )

    word_importance = attributions.abs().sum(dim=2).squeeze(0).detach().cpu().numpy()
    word_importance = (word_importance - word_importance.min()) / (word_importance.max() - word_importance.min() + 1e-8)

    top_indices = np.argsort(word_importance)[-top_k:][::-1]
    influential_words = [tokens[i] for i in top_indices if i < len(tokens)]
    return influential_words


# ============ PREDICT FUNCTION ============
def predict_news(title, body):
    combined = title + " " + body
    sequence, _ = text_to_vectors(combined, w2v, MAX_SEQ_LEN)

    model.eval()
    with torch.no_grad():
        output = model(sequence)
        prob = torch.sigmoid(output).item()

    label = "Real" if prob < 0.5 else "Fake"
    confidence = prob if label == "Fake" else 1 - prob

    if confidence > 0.8:
        verdict = f"Most likely {label}"
    elif confidence > 0.7:
        verdict = f"Might be {label}"
    else:
        verdict = f"Uncertain ({label})"

    similarity = compute_similarity(title, body)
    top_words = get_influential_words_IG(model, combined, w2v, MAX_SEQ_LEN, top_k=10)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "verdict": verdict,
        "similarity": round(similarity, 2),
        "top_words": top_words,
    }
