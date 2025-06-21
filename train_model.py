import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load datasets
fake = pd.read_csv('fake.csv')
true = pd.read_csv('true.csv')

# Add labels: 0 for fake, 1 for true
fake['label'] = 0
true['label'] = 1

# Combine datasets
all_news = pd.concat([fake, true], ignore_index=True)

# Combine title and text
all_news['content'] = all_news['title'].fillna('') + ' ' + all_news['text'].fillna('')

# Features and labels
X = all_news['content']
y = all_news['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print('Model and vectorizer saved!') 