from django.apps import AppConfig

class SentimentAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sentiment_app'

    def ready(self):
        from .service import sentiment_service
        sentiment_service.initialize_sentiment_service()
