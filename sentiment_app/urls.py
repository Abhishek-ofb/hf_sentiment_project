from django.urls import path
from .views import SentimentView, analyze_audio_file


urlpatterns = [
    path('sentiment/', SentimentView.as_view(), name='sentiment'),
    path('analyze-audio/', analyze_audio_file, name='analyze-audio'),
]
