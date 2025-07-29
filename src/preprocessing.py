import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.utils import image_dataset_from_directory
import shutil
import numpy as np


# THIS FUNCTION CHECKS IF THE PATH PROVIDED HAVE THE FOLLOWING STRUCTURE:
# FIRST CHECKS FOR  'test', 'train', 'validate' MAIN FOLDERS
# THEN CHECKS FOR 'bus', 'car', 'motorcycle', 'truck' FOLDERS INSIDE EACH MAIN FOLDER

def check(path):
    required_folders = {'test', 'train', 'validate'}
    required_classes = {'bus', 'truck', 'car', 'motorcycle'}

    if not os.path.isdir(path):
        print("No sub folders validated, train and test found")
        return False

    present_folders = set(
        name.lower() for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )

    if not required_folders.issubset(present_folders):
        print("Not all required sub folders were found")
        return False

    print("All required sub folders found")

    # Check inside each main folder for class subfolders
    for folder in required_folders:
        folder_path = os.path.join(path, folder)
        present_classes = set(
            name.lower() for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))
        )
        if not required_classes.issubset(present_classes):
            print(f"Not all class subfolders found in '{folder}'")
            return False

    print("All class subfolders found in test, train, and validate")
    return True





# DOING DATA AUGMENTATION ON THE DATASET
# AND SAVING THE AUGMENTED IMAGES IN A NEW FOLDER CALLED 'datasetnew'
# SAVED IN THE WORKING DIRECTORY
# THIS FINAL FUNCTION WILL CREATE A NEW DIRECTORY CALLED 'new_dataset'


def aug_set(original_dataset_dir, new_dataset_dir):
    if not check(original_dataset_dir):
        print("Dataset structure invalid. Aborting augmentation.")
        return

    splits = ['train', 'validate', 'test']
    classes = ['bus', 'car', 'motorcycle', 'truck']
    allowed_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    # Check if each class in each split has at least one image
    for split in splits:
        for class_name in classes:
            class_dir = os.path.join(original_dataset_dir, split, class_name)
            if not os.path.isdir(class_dir):
                print(f"Missing directory: {class_dir}")
                print("All classes must have at least an image in each split.")
                return
            has_image = any(
                fname.lower().endswith(allowed_exts)
                for fname in os.listdir(class_dir)
            )
            if not has_image:
                print(f"No images found in: {class_dir}")
                print("All classes must have at least an image in each split.")
                return

    imggen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create directories
    for split in splits:
        for class_name in classes:
            os.makedirs(f"{new_dataset_dir}/{split}/{class_name}", exist_ok=True)

    files_created = 0

    # Augment images (no file removal)
    for split in splits:
        for class_name in classes:
            source_dir = f"{original_dataset_dir}/{split}/{class_name}"
            dest_dir = f"{new_dataset_dir}/{split}/{class_name}"

            for img_file in os.listdir(source_dir):
                if img_file.lower().endswith(allowed_exts):
                    # Copy original
                    shutil.copy2(
                        os.path.join(source_dir, img_file),
                        os.path.join(dest_dir, f"original_{img_file}")
                    )
                    files_created += 1

                    # Generate 4 augmented versions
                    img = load_img(os.path.join(source_dir, img_file))
                    img_array = np.expand_dims(img_to_array(img), 0)
                    aug_iter = imggen.flow(img_array, batch_size=1)
                    for i in range(4):
                        aug_img = array_to_img(next(aug_iter)[0])
                        name = f"aug_{i+1}_{img_file}"
                        aug_img.save(os.path.join(dest_dir, name))
                        files_created += 1

    print(f"Success: files where created.")



