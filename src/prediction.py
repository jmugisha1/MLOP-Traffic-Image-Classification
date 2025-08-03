import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

with open("../class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Class names loaded:", class_names)

def predict_image(model_path, img_path):
    
    model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 

    
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = class_names[pred_idx]
    pred_confidence = preds[0][pred_idx]

    
    print(f"Predicted label: {pred_label}")
    print(f"Confidence: {pred_confidence:.4f}")