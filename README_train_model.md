## `train_model.py` – End‑to‑End Training Script (Deep Dive)

This document explains the purpose and mechanics of every part of `train_model.py`, why it is implemented that way, viable alternatives, and how each piece contributes to training the model. It is meant as an educational deep dive, not just quick start notes.

### What this script does
- Loads labeled text data from CSVs and cleans it.
- Tokenizes text and maps tokens to pretrained Word2Vec embeddings.
- Pads sequences to a uniform length and builds `DataLoader`s.
- Defines a bidirectional GRU classifier for binary classification.
- Trains with early stopping on validation F1.
- Evaluates on test data and saves the best model weights.

---

## 1) Imports and global setup

```python
import os, re, random, numpy as np, pandas as pd, torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
```

- Why: Brings in filesystem (`os`), regex (`re`), randomness control, numerics (`numpy`), dataframes (`pandas`), PyTorch core (`torch`), metrics (`sklearn`), pretrained embeddings (`gensim`), batching utilities (`DataLoader`, `TensorDataset`, `pad_sequence`), and plotting libs.
- Alternatives: You can omit `matplotlib`/`seaborn` if not plotting; replace `gensim` with `fastText`/`spacy`/`torchtext`; use `datasets` (HuggingFace) instead of `pandas`.
- Role: Provides foundational utilities used across the pipeline.

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- Why: Ensures reproducibility and selects GPU when available.
- Alternatives: Also set `torch.cuda.manual_seed_all(SEED)` if using multiple GPUs; control cudnn determinism for stricter reproducibility.
- Role: Stable experiments and hardware utilization.

---

## 2) Dataset paths and loading

```python
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_CSV = os.path.join(BASE_DIR/dataset, "train.csv")
VAL_CSV   = os.path.join(BASE_DIR/dataset, "validation.csv")
TEST_CSV  = os.path.join(BASE_DIR/dataset, "test.csv")
```

- Why: Define absolute paths relative to repository layout.
- Note: In Python, `os.path.join` uses strings; ensure the operand before `/` is a `pathlib.Path` if using `/`. If this is pure `os.path`, write: `os.path.join(BASE_DIR, "dataset", "train.csv")` etc.
- Alternatives: Use `pathlib.Path(__file__).resolve().parents[1] / "dataset" / "train.csv"`.
- Role: Locates the CSVs for train/val/test splits.

```python
def load_csv(path):
    """Load a CSV, clean text columns, and return combined text with labels."""
    df = pd.read_csv(path)
    for col in ["title","text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""
    df["label"] = df["label"].astype(int)
    df["combined"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    return df[["combined","label"]]
```

- Why: Robustly handles missing columns/values, ensures labels are ints, merges `title` and `text` into a single input.
- Alternatives:
  - Use only `text` or concatenate with a separator token.
  - Apply deeper cleaning/normalization here (lowercasing, punctuation handling) instead of later.
- Role: Produces the canonical input (`combined`) and target (`label`).

```python
train_df = load_csv(TRAIN_CSV)
val_df   = load_csv(VAL_CSV)
test_df  = load_csv(TEST_CSV)
print(train_df.head())
```

- Why: Materializes the splits and sanity-checks data shape visually.
- Alternatives: Log via `logging` rather than printing; sample a few randomized rows.
- Role: Prepares dataframes for tokenization/embedding.

---

## 3) Tokenization and embeddings

```python
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

def tokenize(s: str):
    """Tokenize a string into lowercase alphabetic words and apostrophes."""
    return [w.lower() for w in TOKEN_PATTERN.findall(s)]
```

- Why: Simple regex tokenizer that keeps alphabetic tokens and apostrophes (e.g., don't -> "don't").
- Alternatives: `nltk.word_tokenize`, `spacy` tokenizer, BPE/subword tokenizers (Byte-Pair, WordPiece) for neural LMs.
- Role: Produces tokens aligned with word-level embeddings.

```python
W2V_PATH = "./embeddings/GoogleNews-vectors-negative300.bin.gz"
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
EMBED_DIM = w2v.vector_size
```

- Why: Loads pretrained 300-dim GoogleNews Word2Vec embeddings. Using pretrained vectors accelerates convergence and boosts performance with limited labeled data.
- Alternatives: GloVe, fastText, custom Word2Vec, or contextual embeddings (BERT, RoBERTa); for contextual embeddings, the model architecture changes.
- Role: Supplies dense vector representations for tokens.

---

## 4) Convert text to padded sequences

```python
MAX_SEQ_LEN = 100

def text_to_sequence(text, keyed_vectors, max_len):
    """Convert text to a sequence of word vectors (padded/truncated to max_len)."""
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
    """Batch-convert a list/series of texts into a padded tensor of embeddings."""
    sequences = [text_to_sequence(text, keyed_vectors, max_len) for text in texts]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences
```

- Why: Converts token lists into sequences of vectors and pads batches to a consistent shape so they can be stacked into tensors for the RNN.
- Alternatives:
  - Pad/truncate to exact `max_len` for each item rather than dynamic padding across batch.
  - Use an Embedding layer with a vocabulary instead of precomputed vectors; this enables end-to-end training of embeddings.
  - Represent unknown words with random vectors instead of zeros, or learnable `[UNK]`.
- Role: Bridges raw text and model-ready tensors.

```python
X_train = batch_sequences(train_df["combined"], w2v, MAX_SEQ_LEN)
y_train = torch.tensor(train_df["label"].values, dtype=torch.float32)
X_val = batch_sequences(val_df["combined"], w2v, MAX_SEQ_LEN)
y_val = torch.tensor(val_df["label"].values, dtype=torch.float32)
X_test = batch_sequences(test_df["combined"], w2v, MAX_SEQ_LEN)
y_test = torch.tensor(test_df["label"].values, dtype=torch.float32)
print(f"Sequence shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
```

- Why: Materialize tensors for inputs and labels; print shapes for verification.
- Alternatives: Lazy conversion inside a custom `Dataset` to reduce upfront memory; or store only indices and embed on-the-fly.
- Role: Produces the data tensors used by `DataLoader`s.

---

## 5) DataLoaders

```python
BATCH_SIZE = 128

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
```

- Why: Packages tensors into dataset objects and loaders for minibatch iteration; shuffles training data to improve SGD.
- Alternatives: Implement a custom `Dataset` that performs tokenization/embedding on-the-fly; utilize `collate_fn` to pad per batch.
- Role: Efficient batching and iteration in the training loop.

---

## 6) Model: Bidirectional GRU classifier

```python
class GRUClassifier(nn.Module):
    """Bidirectional GRU-based binary classifier over precomputed word embeddings."""
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
        """Encode a sequence batch with GRU and return logits for binary classification."""
        gru_out, h_n = self.gru(x)
        if self.gru.bidirectional:
            h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            h_n = h_n[-1,:,:]
        h_n = self.dropout(h_n)
        logits = self.fc(h_n).squeeze(1)
        return logits
```

- Why: GRUs efficiently summarize sequences. Bidirectionality lets the model use both past and future context.
- Alternatives: LSTM; 1D CNNs; attention pooling; Transformers (fine-tune BERT and use CLS token); mean/max pooling over embeddings.
- Role: Maps a variable-length embedded sequence to a scalar logit per example.

```python
model = GRUClassifier(EMBED_DIM).to(device)
```

- Why: Instantiates the model and moves it to GPU/CPU.
- Alternatives: Replace with chosen architecture.
- Role: Core trainable component.

---

## 7) Training setup and evaluation

```python
LR = 1e-3
EPOCHS = 50
patience = 5

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
```

- Why: Typical defaults for text classification with small models. `BCEWithLogitsLoss` expects raw logits and applies a stable sigmoid internally.
- Alternatives: `AdamW`, `SGD` with momentum; different weight decay; focal loss for class imbalance; cosine annealing or schedulers.
- Role: Defines optimization landscape and objective.

```python
def evaluate(loader):
    """Evaluate model on a DataLoader and compute accuracy, precision, recall, F1."""
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
```

- Why: Centralizes metric computation; uses a 0.5 threshold.
- Alternatives: Tune threshold to maximize F1/ROC-AUC; compute ROC-AUC/PR-AUC.
- Role: Guides early stopping and reports progress.

---

## 8) Main training loop with early stopping

```python
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
    print(" Model saved!")
```

- Why (step-by-step):
  - `model.train()` enables dropout; gradient updates inside the train loop.
  - Zero gradients, forward pass, compute loss, backprop, `optimizer.step()`—the standard SGD iteration.
  - Tracks average train loss per epoch.
  - Validation pass disables grad, computes average val loss; evaluates metrics on train and val.
  - Early stopping monitors `val_f1` to prevent overfitting; stores CPU copy of the best weights.
  - After training, restores best weights, evaluates on test set, and saves the final model.
- Alternatives:
  - Add learning rate schedulers.
  - Monitor validation loss instead of F1 (depending on metric priorities).
  - Use gradient clipping for RNN stability.
  - Mixed precision (`torch.cuda.amp`) for speed/memory.
  - Checkpointing via `torch.save({'epoch':..., 'state_dict':..., ...})`.
- Role: Orchestrates optimization and model selection.

---

## 9) How to run

1. Place Word2Vec file at `./embeddings/GoogleNews-vectors-negative300.bin.gz` (or update `W2V_PATH`).
2. Ensure CSVs at `dataset/train.csv`, `dataset/validation.csv`, `dataset/test.csv` with columns: `title`, `text`, `label` (int 0/1). Missing columns are handled but providing both is recommended.
3. Install deps:

```bash
pip install torch pandas numpy gensim scikit-learn tqdm matplotlib seaborn
```

4. Run the script:

```bash
python -m django-starter.train_model
```

5. Output: console logs of training progress, `gru_classifier.pth` saved weights.

---

## 10) Tuning and extensions

- Sequence length: Increase `MAX_SEQ_LEN` for longer documents; monitor GPU memory and overfitting.
- Batch size: Adjust `BATCH_SIZE` for memory/throughput; too large may degrade generalization.
- Optimizer: Try `AdamW` with weight decay separation; add schedulers (OneCycle, cosine, ReduceLROnPlateau).
- Architecture: Replace GRU with LSTM/Transformer; add attention pooling.
- Embeddings: Switch to trainable `nn.Embedding` with a built vocab; or contextual LMs (BERT family) using `transformers`.
- Class imbalance: Weighted loss (`pos_weight` in `BCEWithLogitsLoss`) or resampling.

---

## 11) Troubleshooting

- All-zero or NaN loss: Check that labels are 0/1 floats; verify embeddings path and shapes.
- Poor validation metrics: Increase data cleaning, adjust tokenizer, tune threshold, try alternative architectures.
- Memory errors: Reduce batch size or seq length; use mixed precision.

---

## 12) Why each part matters for training

- Clean loading and tokenization ensure consistent input semantics.
- Pretrained embeddings inject external knowledge and stabilize learning with fewer labels.
- Padding and batching enable efficient GPU-parallel training.
- Bidirectional GRU captures context, improving classification over naive pooling.
- Proper loss/optimizer and early stopping guide convergence and generalization.
- Final evaluation and saving produce a reproducible artifact for downstream inference.


