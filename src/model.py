from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import  Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory



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

    model = Sequential([
        base_model,                          
        GlobalAveragePooling2D(),           
        Dense(4, activation='softmax')])

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
    model.save('../models/ccn.h5')

    return history


# for images, labels in test_ds.take(1):
#     index = random.randint(0, images.shape[0] - 1)
#     random_image = images[index]
#     random_label = labels[index]
#     break 


# model.save('traffic-image-classification.keras')



# def retrain_model(data, epochs):
#     import tensorflow as tf
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.utils import image_dataset_from_directory
    
#     model = load_model('traffic-image-classification.keras')

#     train_ds = image_dataset_from_directory(data, image_size=(224, 224), batch_size=32)

#     model.fit(train_ds, epochs=epochs)

#     model.save('traffic-image-classification.keras')
#     print("Model retrained and saved!")
    
#     return model

# # how to call the fuction
# # retrained_model = retrain_model('path/to/new/data', 15)



