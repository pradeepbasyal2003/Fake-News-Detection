from django.shortcuts import get_object_or_404, render,redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json, joblib, os
import numpy as np
from django.contrib.auth.decorators import login_required
from .models import NewsArticle,UserFeedback
from .forms import UserFeedbackForm
import gensim
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string

# Custom stopwords for more relevant keywords
CUSTOM_STOPWORDS = set([
    'said', 'will', 'one', 'two', 'new', 'also', 'can', 'may', 'like', 'just', 'get', 'make', 'time', 'year', 'years', 'day', 'days', 'week', 'weeks', 'month', 'months',
    'us', 'mr', 'mrs', 'ms', 'could', 'would', 'should', 'must', 'might', 'still', 'even', 'many', 'much', 'every', 'news', 'report', 'reports', 'say', 'says', 'see', 
    'seen', 'use', 'used', 'using', 'however', 'according', 'including', 'since', 'among', 'within', 'without', 'around', 'across', 'per', 'via', 'due', 'yet', 'another', 
    'others', 'back', 'ago', 'first', 'last', 'next', 'over', 'under', 'before', 'after', 'between', 'through', 'about', 'against', 'above', 'below', 'off', 'on', 'in', 
    'at', 'by', 'to', 'from', 'with', 'for', 'of', 'and', 'or', 'but', 'if', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these',
    'those', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'our', 'your', 'my', 'me', 'you', 'we', 'do', 'does', 'did', 'so', 'no', 'not', 'than', 'then', 'there',
    'here', 'out', 'up', 'down', 'into', 'over', 'again', 'once', 'because', 'while', 'where', 'when', 'who', 'whom', 'which', 'what', 'how', 'why', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
])

# Load models and vectorizer once
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.joblib')
WORD2VEC_PATH = os.path.join(BASE_DIR, 'word2vec.model')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
word2vec_model = gensim.models.Word2Vec.load(WORD2VEC_PATH)

def preprocess_text(text):
    """Tokenize, lowercase, remove punctuation and stopwords."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalpha() and token not in stop_words]

def document_vector(doc, model):
    """Create a document vector by averaging word vectors."""
    processed_tokens = preprocess_text(doc)
    doc_vectors = [model.wv[word] for word in processed_tokens if word in model.wv.key_to_index]
    if not doc_vectors:
        return np.zeros(model.vector_size)
    return np.mean(doc_vectors, axis=0)

def home_view(request):
    return render(request,'home.html')

@csrf_exempt
def predict_fake_news(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error":"unauthenticated"},status=401)
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            title = data.get('title', '')
            body = data.get('body', '')
            
            # --- Prediction using Word2Vec model ---
            content = title + ' ' + body
            content_vec = document_vector(content, word2vec_model).reshape(1, -1)
            pred = model.predict(content_vec)[0]
            proba = model.predict_proba(content_vec)[0]
            confidence = float(np.max(proba))
            
            result = {
                'prediction': 'real' if pred == 1 else 'fake',
                'confidence': confidence
            }
            
            # --- Cosine Similarity using Word2Vec ---
            if title and body:
                title_vec = document_vector(title, word2vec_model).reshape(1, -1)
                text_vec = document_vector(body, word2vec_model).reshape(1, -1)
                similarity_score = cosine_similarity(title_vec, text_vec)[0][0]
                result['similarity'] = f"{similarity_score:.2f} ({similarity_score*100:.0f}% confident match)"

            NewsArticle.objects.create(
                title = title,
                content = body,
                result = "REAL" if pred == 1 else "FAKE",
                confidence_score = confidence,
            )

            # --- Suspicious words using TF-IDF ---
            # Note: We can't get feature importance directly from Word2Vec+LogisticRegression in the same way.
            # We use the TF-IDF vectorizer to find important/suspicious words as planned.
            X_tfidf = vectorizer.transform([content])
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            # Get the scores of the words present in the input
            word_scores = X_tfidf.toarray().flatten()
            word_indices = word_scores.nonzero()[0]
            
            # Create a list of (word, score) tuples
            present_words = [
                (feature_names[i], word_scores[i]) for i in word_indices
                if len(feature_names[i]) > 2 and feature_names[i].lower() not in CUSTOM_STOPWORDS
            ]
            
            # Sort by score and get the top words
            present_words.sort(key=lambda x: -x[1])
            keywords = [word for word, score in present_words[:10]] # Get top 10 suspicious words
            
            if len(keywords) > 0:
                result['keywords'] = keywords
                
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return redirect("/")

# Create your views here.


def submit_feedback(request):
    

    if request.method == 'POST':
        form = UserFeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save(commit=False)
            feedback.user = request.user
            
            feedback.save()
            return redirect('/')
    else:
        form = UserFeedbackForm()

    return render(request, 'feedback_form.html', {'form': form})