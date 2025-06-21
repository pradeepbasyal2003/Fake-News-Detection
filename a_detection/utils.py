import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Explanation: Download NLTK resources if not already present.
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource.startswith('punkt') else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

def preprocess_text(text):
    """Tokenize, lowercase, remove punctuation and stopwords."""
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords if you have any
    # stop_words.update(CUSTOM_STOPWORDS) 
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalpha() and token not in stop_words] 