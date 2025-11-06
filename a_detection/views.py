from django.shortcuts import get_object_or_404, render,redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json, joblib, os
import numpy as np
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import NewsArticle,UserFeedback
from .forms import UserFeedbackForm
import gensim
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
from .utils import preprocess_text
from .model_loader import predict
from a_core.settings import NEWS_API_KEY
import httpx
import spacy
from collections import Counter
# Custom stopwords for more relevant keywords



def home_view(request):
    return render(request,'home.html')

@csrf_exempt
def predict_fake_news(request):
    if not request.user.is_authenticated:
        return JsonResponse({
            "error": "unauthenticated",
            "message": "Please log in to use the fake news detection feature."
        }, status=401)
    if request.method == 'POST':
            data = json.loads(request.body)
            title = data.get('title', '')
            body = data.get('body', '')
            print(f"title:{title}")
            result = predict(title, body)
            print(result)
            
            NewsArticle.objects.create(
                title = title,
                content = body,
                result = "REAL" if result["prediction"] == "REAL" else "FAKE",
                confidence_score = result["confidence"],
            )

            recommended_articles = get_real_news_recommendations(title)
            result["article recommendations"] = recommended_articles
            return JsonResponse(result)
    else:
        return redirect("/")




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



def get_real_news_recommendations(title : str):
    """
    Returns a list of real news articles that are similar to the given title.
    """
    query = get_boarder_query(title)

    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&pageSize=5&apiKey={NEWS_API_KEY}"

    client = httpx.Client()
    response = client.get(url)
    print(f"response :: {response.text}")
    data = response.json()

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "title": a["title"],
            "source": a["source"]["name"],
            "url": a["url"],
            "description": a["description"]
        })
    print(articles)
    return articles


def get_boarder_query(title:str):
    """
    Returns a boarder query after filtering unnecessary words.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(title)

    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    broad_query = " ".join(keywords)
    print(broad_query)
    return broad_query