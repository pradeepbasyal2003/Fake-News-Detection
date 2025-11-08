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
from urllib.parse import urlparse
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



@csrf_exempt
def get_real_news_recommendations(request):
    """
    View function that returns a list of real news articles similar to the given title.
    Accepts POST with JSON body containing 'title' field, or GET with 'title' query parameter.
    """
    TRUSTED_SOURCES = [
    # Global
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "associatedpress.com",
    "theguardian.com", "nytimes.com", "cnn.com", "aljazeera.com", "npr.org",
    "bloomberg.com", "forbes.com", "theconversation.com", "politico.com",
    "time.com", "usatoday.com", "theatlantic.com", "washingtonpost.com", "wsj.com",

    # U.S.
    "abcnews.go.com", "cbsnews.com", "nbcnews.com", "latimes.com",
    "chicagotribune.com", "pbs.org", "vox.com", "thehill.com", "propublica.org",

    # U.K.
    "independent.co.uk", "telegraph.co.uk", "mirror.co.uk", "economist.com", "sky.com",

    # Canada
    "cbc.ca", "globalnews.ca", "ctvnews.ca", "torontosun.com", "nationalpost.com",

    # Europe
    "dw.com", "euronews.com", "lemonde.fr", "spiegel.de", "rtbf.be",

    # Science / Tech
    "techcrunch.com", "wired.com", "arstechnica.com", "engadget.com",
    "scientificamerican.com", "nature.com", "nationalgeographic.com", "newscientist.com"
]
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            title = data.get('title', '')
        except json.JSONDecodeError:
            return JsonResponse({
                "error": "invalid_json",
                "message": "Invalid JSON in request body"
            }, status=400)
    elif request.method == 'GET':
        title = request.GET.get('title', '')
    else:
        return JsonResponse({
            "error": "method_not_allowed",
            "message": "Only GET and POST methods are allowed"
        }, status=405)
    
    if not title:
        return JsonResponse({
            "error": "missing_title",
            "message": "Title parameter is required"
        }, status=400)
    
    try:
        query = get_boarder_query(title)
        # Increase pageSize to get more results for filtering
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&pageSize=50&apiKey={NEWS_API_KEY}"
        
        client = httpx.Client()
        response = client.get(url)
        print(f"response :: {response.text}")
        data = response.json()
        
        # Normalize trusted sources to lowercase for comparison
        trusted_sources_lower = [source.lower() for source in TRUSTED_SOURCES]
        
        # Helper function to extract domain from URL
        def extract_domain(url):
            """Extract domain from URL (e.g., 'https://www.bbc.com/news' -> 'bbc.com')"""
            if not url:
                return ""
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                # Remove 'www.' prefix if present
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
            except:
                return ""
        
        # Helper function to check if article is from trusted source
        def is_trusted_source(article):
            """Check if article is from a trusted source by checking URL domain and source name"""
            source_name = article.get("source", {}).get("name", "").lower()
            article_url = article.get("url", "")
            domain = extract_domain(article_url)
            
            # Check if domain matches any trusted source
            for trusted in trusted_sources_lower:
                if trusted in domain or domain in trusted:
                    return True
            
            # Check if source name matches (normalize common variations)
            source_name_normalized = source_name.replace(" ", "").replace(".", "")
            for trusted in trusted_sources_lower:
                trusted_normalized = trusted.replace(".", "")
                # Check exact match or if trusted source is in source name
                if trusted_normalized in source_name_normalized or source_name_normalized in trusted_normalized:
                    return True
                # Check common name variations (e.g., "BBC" matches "bbc.com")
                if trusted.startswith("bbc") and ("bbc" in source_name_normalized):
                    return True
                if trusted.startswith("reuters") and ("reuters" in source_name_normalized):
                    return True
                if trusted.startswith("cnn") and ("cnn" in source_name_normalized):
                    return True
                if trusted.startswith("nytimes") and ("new york times" in source_name or "nytimes" in source_name_normalized):
                    return True
                if trusted.startswith("theguardian") and ("guardian" in source_name_normalized):
                    return True
                if trusted.startswith("aljazeera") and ("al jazeera" in source_name or "aljazeera" in source_name_normalized):
                    return True
                if trusted.startswith("npr") and ("npr" in source_name_normalized):
                    return True
                if trusted.startswith("cbc") and ("cbc" in source_name_normalized):
                    return True
            
            return False
        
        # Filter articles to only include trusted sources
        articles = []
        for a in data.get("articles", []):
            if is_trusted_source(a):
                articles.append({
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "description": a.get("description", "")
                })
        
        # Limit to top 5 trusted articles
        articles = articles[:5]
        
        print(f"Filtered articles from trusted sources: {len(articles)}")
        return JsonResponse({
            "success": True,
            "articles": articles,
            "count": len(articles)
        })
    except Exception as e:
        print(f"Error fetching news recommendations: {str(e)}")
        return JsonResponse({
            "error": "server_error",
            "message": f"Failed to fetch news recommendations: {str(e)}"
        }, status=500)


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


