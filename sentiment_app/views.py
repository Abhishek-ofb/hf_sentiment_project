from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
import tempfile, os

from .serializers import TextInputSerializer
from .service import pipeline_service

import os
import json
import tempfile
import requests
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from .service import pipeline_service


EXOTEL_BASE_URL = "https://recordings.exotel.com/exotelrecordings/ofbusiness2/"
EXOTEL_AUTH_HEADER = "Basic NTFlMjM3N2EwMTJjNTAxNGU0ZjE5ODVhMzg1NWRmODEzMWY2OWMxY2I0YTY1YTY1OmQ4MzdhZGFjYzRiNjFjMWY4ZmFjYjQ4ZjNhN2UwZjFmNzUxYTVmZDU2NWZlMzNlMA=="

class SentimentView(APIView):
    """For raw text input sentiment analysis"""

    def post(self, request):
        serializer = TextInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        text = serializer.validated_data["text"]
        try:
            result = pipeline_service.process_text(text)
            return Response(result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@csrf_exempt
def analyze_audio_file(request: HttpRequest):
    """Handles audioFileName → downloads recording → transcription → translation → sentiment + summary"""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are accepted"}, status=405)

    try:
        # Expect JSON body: { "audioFileName": "filename.mp3" }
        data = request.POST or request.body
        if not data:
            return JsonResponse({"error": "No request data provided"}, status=400)

        if isinstance(data, (bytes, str)):
            data = json.loads(data)

        audio_file_name = data.get("audioFileName")
        if not audio_file_name:
            return JsonResponse({"error": "Missing audioFileName"}, status=400)

        # Build download URL
        audio_url = f"{EXOTEL_BASE_URL}{audio_file_name}"

        # Download from Exotel
        headers = {"Authorization": EXOTEL_AUTH_HEADER}
        response = requests.get(audio_url, headers=headers, stream=True)

        if response.status_code != 200:
            return JsonResponse({"error": "Failed to download audio", "status": response.status_code}, status=500)

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        # Process audio → transcription, translation, sentiment, summary
        result = pipeline_service.process_audio(temp_file_path)

        # Cleanup
        os.remove(temp_file_path)

        return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse(
            {"error": "Processing failed", "details": str(e)},
            status=500
        )
