from rest_framework.views import APIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from tensorflow.keras.models import load_model
from rest_framework import status
from PIL import Image
import numpy as np



# FUNCTION TO MAKE A PREDICTION USING THE MODEL
model_path = r'C:/Users/Administrator\Desktop/Projects/MLOP-Traffic-Image-Classification/models/image-classes.h5'
model = load_model(model_path)

class_names = ['bus', 'car', 'motorcycle', 'truck']

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided.'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            image_file = request.FILES['image']
            img = Image.open(image_file).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            pred_idx = int(np.argmax(preds, axis=1)[0])
            pred_label = class_names[pred_idx]
            pred_confidence = float(preds[0][pred_idx])

            # Print to server console
            print(f"Predicted class: {pred_label}, Confidence: {pred_confidence:.4f}")

            return Response({
                'predicted_class': pred_label,
                'confidence': pred_confidence
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

