from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import TextInputSerializer
from . import sentiment_service

class SentimentView(APIView):
    def post(self, request):
        serializer = TextInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        text = serializer.validated_data["text"]
        try:
            result = sentiment_service.analyze_text(text)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "text": text,
            "sentiment": result.get("label"),
            "score": result.get("score"),
            "raw": result.get("raw")
        })
