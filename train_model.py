import os, re, random, numpy as np, pandas as pd, torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Reproducibility & device
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CSV paths (LOCAL)
# -----------------------------
BASE = "./data"
TRAIN_CSV = os.path.join(BASE, "train.csv")
VAL_CSV   = os.path.join(BASE, "validation.csv")
TEST_CSV  = os.path.join(BASE, "test.csv")

# -----------------------------
# Load CSVs
# -----------------------------
def load_csv(path):
    df = pd.read_csv(path)
    for col in ["title","text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""
    df["label"] = df["label"].astype(int)
    df["combined"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    return df[["combined","label"]]

train_df = load_csv(TRAIN_CSV)
val_df   = load_csv(VAL_CSV)
test_df  = load_csv(TEST_CSV)
print(train_df.head())

# -----------------------------
# Tokenizer
# -----------------------------
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

def tokenize(s: str):
    return [w.lower() for w in TOKEN_PATTERN.findall(s)]

# -----------------------------
# Load Word2Vec (LOCAL)
# -----------------------------
W2V_PATH = "./embeddings/GoogleNews-vectors-negative300.bin.gz"

print("Loading Word2Vec ...")
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
EMBED_DIM = w2v.vector_size

# -----------------------------
# Convert text to embedding sequences
# -----------------------------
MAX_SEQ_LEN = 100

def text_to_sequence(text, keyed_vectors, max_len):
    tokens = tokenize(text)[:max_len]
    vectors = []
    for token in tokens:
        if token in keyed_vectors:
            vectors.append(keyed_vectors[token])
        else:
            vectors.append(np.zeros(keyed_vectors.vector_size, dtype=np.float32))
    if vectors:
        return torch.tensor(np.array(vectors), dtype=torch.float32)
    else:
        return torch.zeros((1, keyed_vectors.vector_size), dtype=torch.float32)

def batch_sequences(texts, keyed_vectors, max_len):
    sequences = [text_to_sequence(text, keyed_vectors, max_len) for text in texts]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences

X_train = batch_sequences(train_df["combined"], w2v, MAX_SEQ_LEN)
y_train = torch.tensor(train_df["label"].values, dtype=torch.float32)

X_val = batch_sequences(val_df["combined"], w2v, MAX_SEQ_LEN)
y_val = torch.tensor(val_df["label"].values, dtype=torch.float32)

X_test = batch_sequences(test_df["combined"], w2v, MAX_SEQ_LEN)
y_test = torch.tensor(test_df["label"].values, dtype=torch.float32)

print(f"Sequence shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

# -----------------------------
# DataLoaders
# -----------------------------
BATCH_SIZE = 128

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# GRU Classifier
# -----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, num_layers=1, bidirectional=True):
        super().__init__()
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
        logits = self.fc(h_n).squeeze(1)
        return logits

model = GRUClassifier(EMBED_DIM).to(device)

# -----------------------------
# Training setup
# -----------------------------
LR = 1e-3
EPOCHS = 50
patience = 5

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).long().cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(yb.long().cpu().numpy().tolist())
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return acc, p, r, f1

# -------------------------------------------------------
# ✅ MAIN RUN BLOCK (Required for VS Code)
# -------------------------------------------------------
if __name__ == "__main__":

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_f1, best_state = -1, None
    early_stop_counter = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_epoch += loss.item() * xb.size(0)
        val_loss_epoch /= len(val_ds)
        val_losses.append(val_loss_epoch)

        train_acc, train_p, train_r, train_f1 = evaluate(train_loader)
        val_acc, val_p, val_r, val_f1 = evaluate(val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: train_F1={train_f1:.3f}  val_F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test
    test_acc, test_p, test_r, test_f1 = evaluate(test_loader)
    print("\nTEST RESULTS:")
    print(test_acc, test_p, test_r, test_f1)

    torch.save(model.state_dict(), "gru_classifier.pth")
    print("✅ Model saved!")
