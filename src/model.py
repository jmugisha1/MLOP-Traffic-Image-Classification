from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import  Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model, Model  
import matplotlib.pyplot as plt
import numpy as np
import random



def cnn_model( train_dir, validate_dir, test_dir):


    # loading the dataset
    train_ds = image_dataset_from_directory(train_dir, image_size=(224, 224), batch_size=32, label_mode='categorical')
    validate_ds = image_dataset_from_directory(validate_dir, image_size=(224, 224), batch_size=32, label_mode='categorical')
    test_ds = image_dataset_from_directory(test_dir, image_size=(224, 224), batch_size=32, label_mode='categorical')

    # getting class names and save it in .txt file
    class_names = train_ds.class_names
    with open("../class_names.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    # Normalize pixel values to be between 0 and 1
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    validate_ds = validate_ds.map(lambda x, y: (x / 255.0, y))
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

    base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
    
    base_model.trainable = False 

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False) 
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(4, activation='softmax')(x)
    model = Model(inputs, outputs)


    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=validate_ds,
        batch_size=32,
        epochs=5,
        verbose=2)
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    
    # Save the model
    model.save('C:/Users/Administrator/Desktop/Projects/MLOP-Traffic-Image-Classification/models/image-classes.h5')

    return history


# PREDICTING A RANDOM IMAGE IN TEST DATASET

# FUNCTION TO RETRAIN THE MODEL

def retrain_model(train_dir):
    
    train_ds = image_dataset_from_directory(train_dir, image_size=(224, 224), batch_size=32, label_mode='categorical')
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))

    
    model = load_model('C:/Users/Administrator/Desktop/Projects/MLOP-Traffic-Image-Classification/models/image-classes.h5')

    # 
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 
    history = model.fit(train_ds, epochs=5)

    # Save the model
    model.save('C:/Users/Administrator/Desktop/Projects/MLOP-Traffic-Image-Classification/models/image-classes.h5')

    print("Model retrained and saved!")
    # Print only the final results
    print("Final training results:")
    for key in history.history:
        print(f"{key}: {history.history[key][-1]}")
    
    return history