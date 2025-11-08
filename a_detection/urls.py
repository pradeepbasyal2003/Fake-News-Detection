from django.urls import path
from .views import *

urlpatterns = [
    path('', home_view, name='home'),
    path('api/predict/', predict_fake_news, name='predict_fake_news'),
    path('feedback/', submit_feedback, name='submit_feedback'),
    path('get-news-recommendations/',get_real_news_recommendations, name='news-recommendations'),
] 