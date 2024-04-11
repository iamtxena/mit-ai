"""
This module is designed for predicting emotions from facial images using a pre-trained model.
It includes functions for making predictions on individual images or batches of images and
plotting the results for a subset of the test dataset.
"""

import os
import random
from datetime import datetime
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model
from tensorflow.keras.preprocessing.image import DirectoryIterator


def predict_and_plot(
    model: Model,
    test_data: Union[np.ndarray, DirectoryIterator],
    y_test_encoded: Optional[pd.DataFrame],
    CATEGORIES: List[str],
) -> str:
    """
    Predicts emotions for a set of images and plots the results.

    This function takes a pre-trained model, test dataset (either as a numpy array or a generator),
    encoded test labels (optional), and a list of category names. It randomly selects an image,
    predicts its emotion using the model, and plots both the predicted and actual labels.
    The plot is saved to a file in the 'results' directory with a timestamped filename.

    Parameters:
    - model (Model): A pre-trained Keras model for emotion prediction.
    - test_data (Union[np.ndarray, DirectoryIterator]): The test dataset, either as a numpy array or a generator.
    - y_test_encoded (Optional[pd.DataFrame]): The encoded labels for the test dataset (optional).
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
        if isinstance(test_data, np.ndarray):
            # Current way: test_data is X_test (numpy array)
            if len(test_data) == 0:
                raise ValueError("test_data is empty. Cannot generate random index.")
            random_index = random.randint(0, len(test_data) - 1)
            random_image = test_data[random_index]
            if y_test_encoded is not None:
                actual_label_index = (
                    np.argmax(y_test_encoded.values[random_index])
                    if isinstance(y_test_encoded, pd.DataFrame)
                    else np.argmax(y_test_encoded[random_index])
                )
                actual_label = CATEGORIES[actual_label_index]
            else:
                actual_label = "Unknown"
        else:
            # New way: test_data is test_generator
            random_index = random.randint(0, len(test_data) - 1)
            random_image = test_data[random_index][0][0]  # Get the first image from the batch
            actual_label_index = test_data.classes[test_data.index_array[random_index * test_data.batch_size]]
            actual_label = CATEGORIES[actual_label_index]

        # Predict the emotion of the image using the provided model
        predicted_probabilities = model.predict(np.expand_dims(random_image, axis=0))
        predicted_label_index = np.argmax(predicted_probabilities)
        predicted_label = CATEGORIES[predicted_label_index]

        # Plot the image in the corresponding subplot
        ax = axes[i]
        # Assuming X_test is 4D with the last dimension being 1 for grayscale
        ax.imshow(random_image.squeeze(), cmap="gray")
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


def predict_images(model: Model, images: Union[np.ndarray, DirectoryIterator], CATEGORIES: List[str]) -> List[str]:
    """
    Predicts emotions for a given array of images or a generator using the provided model.

    Parameters:
    - model (Model): A pre-trained Keras model for emotion prediction.
    - images (Union[np.ndarray, DirectoryIterator]): The dataset, either as a numpy array or a generator.
    - CATEGORIES (List[str]): A list of category names corresponding to the encoded labels.

    Returns:
    - List[str]: A list of predicted labels for each image.
    """
    predicted_labels = []
    if isinstance(images, np.ndarray):
        # Current way: images is a numpy array
        for image in images:
            predicted_probabilities = model.predict(np.expand_dims(image, axis=0))
            predicted_label_index = np.argmax(predicted_probabilities)
            predicted_label = CATEGORIES[predicted_label_index]
            predicted_labels.append(predicted_label)
    else:
        # New way: images is a generator
        for batch in images:
            predicted_probabilities = model.predict(batch[0])  # Predict on the batch of images
            predicted_label_indices = np.argmax(predicted_probabilities, axis=1)
            predicted_batch_labels = [CATEGORIES[index] for index in predicted_label_indices]
            predicted_labels.extend(predicted_batch_labels)

    return predicted_labels
