import os
import re
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from captum.attr import IntegratedGradients
from nltk.corpus import stopwords
import nltk

# -----------------------------
# Ensure stopwords are available
# -----------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# -----------------------------
# Paths & config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "gru_classifier.pth")
W2V_PATH = os.path.join(BASE_DIR, "GoogleNews-vectors-negative300.bin.gz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 100
EMBED_DIM = 300

# -----------------------------
# GRU model (matches training)
# -----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, num_layers=1, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        if self.gru.bidirectional:
            h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            h_n = h_n[-1,:,:]
        h_n = self.dropout(h_n)
        logits = self.fc(h_n)
        return torch.sigmoid(logits)

# -----------------------------
# Preload models
# -----------------------------
print("Loading Word2Vec and GRU model...")
_w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
_model = GRUClassifier(EMBED_DIM)
_state_dict = torch.load(MODEL_PATH, map_location=device)
_model.load_state_dict(_state_dict)
_model.eval()
_model.to(device)
print(" Models loaded successfully")

# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = [t for t in text.split() if t not in stop_words]
    return tokens

def text_to_tensor(tokens, max_len=MAX_SEQ_LEN):
    vectors = [_w2v[t] if t in _w2v else np.zeros(EMBED_DIM) for t in tokens]
    vectors = vectors[:max_len]
    while len(vectors) < max_len:
        vectors.append(np.zeros(EMBED_DIM))
    return torch.tensor([vectors], dtype=torch.float32).to(device)

# -----------------------------
# Cosine similarity
# -----------------------------
def compute_similarity(title, content):
    t_vec = text_to_tensor(preprocess_text(title))
    b_vec = text_to_tensor(preprocess_text(content))
    t_mean = t_vec.mean(dim=1).cpu().numpy()
    b_mean = b_vec.mean(dim=1).cpu().numpy()
    return float(cosine_similarity(t_mean, b_mean)[0][0])

# -----------------------------
# Integrated Gradients
# -----------------------------
def get_influential_words_IG(text, top_k=10):
    tokens = preprocess_text(text)
    vectors = np.array([_w2v[t] if t in _w2v else np.zeros(EMBED_DIM) for t in tokens])
    if len(vectors) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(vectors)
        vectors = np.vstack([vectors, np.zeros((pad_len, EMBED_DIM))])
    else:
        vectors = vectors[:MAX_SEQ_LEN]

    sequence = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).to(device)
    ig = IntegratedGradients(_model)
    attributions, _ = ig.attribute(sequence, baselines=torch.zeros_like(sequence), return_convergence_delta=True)
    word_importance = attributions.abs().sum(dim=2).squeeze(0).detach().cpu().numpy()
    word_importance = (word_importance - word_importance.min()) / (word_importance.max() - word_importance.min() + 1e-8)
    top_indices = np.argsort(word_importance)[-top_k:][::-1]
    return [tokens[i] for i in top_indices if i < len(tokens)]

# -----------------------------
# Predict function
# -----------------------------
def predict(title, content):
    combined = f"{title} {content}"
    tokens = preprocess_text(combined)
    x = text_to_tensor(tokens)

    with torch.no_grad():
        prob = _model(x).item()

    # Confidence corresponds to predicted label
    if prob > 0.5:
        label = "REAL"
        confidence = prob
    else:
        label = "FAKE"
        confidence = 1 - prob

    # Human-readable verdict
    if confidence > 0.8:
        verdict = f"Most likely {label}"
    elif confidence > 0.7:
        verdict = f"Might be {label}"
    else:
        verdict = f"Uncertain ({label})"

    similarity = compute_similarity(title, content)
    top_words = get_influential_words_IG(combined)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "verdict": verdict,
        "similarity": round(similarity, 2),
        "top_words": top_words
    }

# -----------------------------
# Optional test run
# -----------------------------
if __name__ == "__main__":
    example = predict("Scientists confirm hot water prevents COVID", 
                      "Drink hot water every 15 minutes...")
    print(example)
