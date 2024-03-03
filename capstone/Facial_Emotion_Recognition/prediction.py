"""
This module is designed for predicting emotions from facial images using a pre-trained model.
It includes functions for making predictions on individual images or batches of images and
plotting the results for a subset of the test dataset.
"""

import os
import random
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model


def predict_and_plot(model: Model, X_test: np.ndarray, y_test_encoded: pd.DataFrame, CATEGORIES: List[str]) -> str:
    """
    Predicts emotions for a set of images and plots the results.

    This function takes a pre-trained model, test dataset, encoded test labels, and a list of category names.
    It randomly selects a subset of images, predicts their emotions using the model, and plots both the
    predicted and actual labels in a 2x2 grid. The plot is saved to a file in the 'results' directory with
    a timestamped filename.

    Parameters:
    - model (Model): A pre-trained Keras model for emotion prediction.
    - X_test (np.ndarray): The test dataset consisting of images.
    - y_test_encoded (pd.DataFrame or np.ndarray): The encoded labels for the test dataset.
    - CATEGORIES (List[str]): A list of category names corresponding to the encoded labels.

    Returns:
    - str: The filename of the saved plot, excluding the 'results' directory path.
    """
    # Set the number of images to display
    num_images = 4

    # Create a figure with subplots in a 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i in range(num_images):
        # Pick a random image from X_test
        random_index = random.randint(0, len(X_test) - 1)
        test_image = X_test[random_index]

        # Predict the emotion of the image using the provided model
        predicted_probabilities = model.predict(np.expand_dims(test_image, axis=0))
        predicted_label_index = np.argmax(predicted_probabilities)
        predicted_label = CATEGORIES[predicted_label_index]

        # Retrieve the actual label for the image
        actual_label_index = (
            np.argmax(y_test_encoded.values[random_index])
            if isinstance(y_test_encoded, pd.DataFrame)
            else np.argmax(y_test_encoded[random_index])
        )
        actual_label = CATEGORIES[actual_label_index]

        # Plot the image in the corresponding subplot
        ax = axes[i]
        # Assuming X_test is 4D with the last dimension being 1 for grayscale
        ax.imshow(test_image.squeeze(), cmap="gray")
        ax.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
        ax.axis("off")  # Hide the axis

    plt.tight_layout()

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Define the results directory and ensure it exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the figure to the results directory with a timestamp
    plot_filename = f"prediction_{timestamp}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free memory

    # Return the path without the 'results' directory
    return plot_filename


def predict_images(model: Model, images: np.ndarray, CATEGORIES: List[str]) -> List[str]:
    """
    Predicts emotions for a given array of images using the provided model.

    Parameters:
    - model (Model): A pre-trained Keras model for emotion prediction.
    - images (np.ndarray): The dataset consisting of images to predict.
    - CATEGORIES (List[str]): A list of category names corresponding to the encoded labels.

    Returns:
    - List[str]: A list of predicted labels for each image.
    """
    predicted_labels = []
    for image in images:
        # Predict the emotion of the image using the provided model
        predicted_probabilities = model.predict(np.expand_dims(image, axis=0))
        predicted_label_index = np.argmax(predicted_probabilities)
        predicted_label = CATEGORIES[predicted_label_index]
        predicted_labels.append(predicted_label)

    return predicted_labels
