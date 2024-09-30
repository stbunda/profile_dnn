import os
from PIL import Image
import numpy as np


def load_images_from_directory(directory, image_size=(256, 256), color_mode="grayscale", class_names=None):
    images = []
    labels = []

    # Map class names to indices
    class_to_index = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)

            # Convert image to grayscale if needed
            if color_mode == "grayscale":
                img = img.convert("L")

            # Resize image
            img = img.resize(image_size)

            # Convert image to numpy array
            img_array = np.array(img)

            # Normalize pixel values to range [0, 1]
            # img_array = img_array / 255.0

            # Append the image and label to the list
            images.append(img_array)
            labels.append(class_to_index[class_name])

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode labels if needed
    labels = np.eye(len(class_names))[labels]

    return images, labels


def batch_generator(images, labels, batch_size):
    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size], labels[i:i+batch_size]