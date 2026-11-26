import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Number of times to train the neural network
EPOCHS = 10

# Image dimensions
IMG_WIDTH = 30
IMG_HEIGHT = 30

# Number of categories
NUM_CATEGORIES = 43

# Fraction of data to use for testing
TEST_SIZE = 0.4


def main():
    """Load data, train the model, and evaluate its performance."""
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and their corresponding labels
    images, labels = load_data(sys.argv[1])

    # Convert labels to one-hot encoded vectors
    labels = tf.keras.utils.to_categorical(labels, NUM_CATEGORIES)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images),
        np.array(labels),
        test_size=TEST_SIZE,
        random_state=42
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file, if specified
    if len(sys.argv) == 3:
        model.save(sys.argv[2])


def load_data(data_dir):
    """Load traffic sign images and their category labels from data_dir."""
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_dir):
            continue

        for filename in sorted(os.listdir(category_dir)):
            image_path = os.path.join(category_dir, filename)
            image = cv2.imread(image_path)

            # Skip files that OpenCV cannot read as images
            if image is None:
                continue

            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images.append(image)
            labels.append(category)

    return images, labels


def get_model():
    """Create and return a compiled convolutional neural network."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
