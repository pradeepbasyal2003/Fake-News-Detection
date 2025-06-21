from django.shortcuts import render,redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json, joblib, os
import numpy as np
from django.contrib.auth.decorators import login_required
from .models import NewsArticle

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

# Load model and vectorizer once
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vectorizer.joblib')
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

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
            content = title + ' ' + body
            X_vec = vectorizer.transform([content])
            pred = model.predict(X_vec)[0]
            proba = model.predict_proba(X_vec)[0]
            confidence = float(np.max(proba))
            result = {
                'prediction': 'real' if pred == 1 else 'fake',
                'confidence': confidence
            }
            NewsArticle.objects.create(
                title = title,
                content = body,
                result = "REAL" if pred ==1 else "FAKE",
                confidence_score = confidence,
            )
            # Feature importance: get top words for the predicted class
            feature_names = np.array(vectorizer.get_feature_names_out())
            coefs = model.coef_[0]
            input_indices = X_vec.nonzero()[1]
            input_words = feature_names[input_indices]
            input_coefs = coefs[input_indices]
            # Filter out stopwords and short words
            filtered = [
                (w, c) for w, c in zip(input_words, input_coefs)
                if len(w) > 2 and w.lower() not in CUSTOM_STOPWORDS
            ]
            if pred == 0:
                # Only show words with significant positive contribution
                filtered = [(w, c) for w, c in filtered if c > 0.05]
                filtered.sort(key=lambda x: -x[1])
                keywords = [w for w, c in filtered[:20]]
            else:
                filtered = [(w, c) for w, c in filtered if c < -0.05]
                filtered.sort(key=lambda x: x[1])
                keywords = [w for w, c in filtered[:5]]
            if len(keywords) > 0:
                result['keywords'] = keywords
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return redirect("/")

# Create your views here.
