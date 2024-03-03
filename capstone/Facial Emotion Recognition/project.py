"""
This module is part of a capstone project for Facial Emotion Recognition. 
It includes the definition of the FacialEmotionRecognition class, which is responsible for loading, 
preprocessing, and managing the dataset for training, validation, and testing purposes.
"""

import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import yaml
from model import run_model
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


class FacialEmotionRecognition:
    """
    A class for recognizing facial emotions from images.

    Attributes:
        data_dir (str): Directory where the data is stored.
        categories (List[str]): List of categories or emotions to recognize.
        subdirs_dict (Dict[str, str]): Dictionary mapping data subsets to their directory names.
        X_train, X_validation, X_test (List[np.ndarray]): Lists to store training, validation, and test images.
        y_train, y_validation, y_test (List[int]): Lists to store labels for training, validation, and test images.
        y_train_df, y_validation_df, y_test_df (pd.DataFrame): DataFrames to store labels for training, validation, and test datasets.
        y_train_encoded, y_validation_encoded, y_test_encoded (List[int]): Lists to store one-hot encoded labels for training, validation, and test datasets.
    """

    def __init__(self, data_dir: str, categories: List[str]):
        self.data_dir = data_dir
        self.categories = categories
        self.subdirs_dict = {"train": "train", "validation": "validation", "test": "test"}
        self.X_train, self.y_train = [], []
        self.X_validation, self.y_validation = [], []
        self.X_test, self.y_test = [], []
        self.y_train_df, self.y_validation_df, self.y_test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.y_train_encoded, self.y_validation_encoded, self.y_test_encoded = [], [], []

    def load_data(self) -> None:
        """
        Loads images and their labels from the dataset directories.
        """
        for key, subdir in self.subdirs_dict.items():
            images, labels = [], []
            for category in self.categories:
                path = os.path.join(self.data_dir, subdir, category)
                class_num = self.categories.index(category)
                for img_name in os.listdir(path):
                    try:
                        img_path = os.path.join(path, img_name)
                        img = Image.open(img_path)
                        images.append(np.array(img))
                        labels.append(class_num)
                    except (IOError, ValueError) as e:
                        print(f"Failed to process {img_name}: {e}")
            if key == "train":
                self.X_train, self.y_train = images, labels
            elif key == "validation":
                self.X_validation, self.y_validation = images, labels
            elif key == "test":
                self.X_test, self.y_test = images, labels

    def preprocess_data(self) -> None:
        """
        Preprocesses the data by converting image lists to numpy arrays, normalizing pixel values,
        shuffling the data, creating label DataFrames, and one-hot encoding the labels.
        """
        # Convert lists to numpy arrays and add an extra dimension for grayscale channel
        self.X_train = np.expand_dims(np.array(self.X_train), axis=-1) / 255.0
        self.X_validation = np.expand_dims(np.array(self.X_validation), axis=-1) / 255.0
        self.X_test = np.expand_dims(np.array(self.X_test), axis=-1) / 255.0
        self.y_train = np.array(self.y_train)
        self.y_validation = np.array(self.y_validation)
        self.y_test = np.array(self.y_test)

        # Shuffle the data
        self.shuffle_data()

        # Create DataFrames for labels
        self.create_label_dataframes()

        # One-hot encode the labels
        self.one_hot_encode_labels()

    def shuffle_data(self) -> None:
        """
        Shuffles the training, validation, and test datasets.
        """
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)
        self.X_train, self.y_train = self.X_train[indices], self.y_train[indices]

        indices = np.arange(self.X_validation.shape[0])
        np.random.shuffle(indices)
        self.X_validation, self.y_validation = self.X_validation[indices], self.y_validation[indices]

        indices = np.arange(self.X_test.shape[0])
        np.random.shuffle(indices)
        self.X_test, self.y_test = self.X_test[indices], self.y_test[indices]

    def create_label_dataframes(self) -> None:
        """
        Creates pandas DataFrames for training, validation, and test labels.
        These DataFrames are used for easier manipulation and analysis of the labels.
        """
        self.y_train_df = pd.DataFrame(self.y_train, columns=["Label"])
        self.y_validation_df = pd.DataFrame(self.y_validation, columns=["Label"])
        self.y_test_df = pd.DataFrame(self.y_test, columns=["Label"])

        # Map the numerical labels to actual emotion classes
        index_to_emotion = {i: emotion for i, emotion in enumerate(self.categories)}
        self.y_train_df["Label"] = self.y_train_df["Label"].map(index_to_emotion)
        self.y_validation_df["Label"] = self.y_validation_df["Label"].map(index_to_emotion)
        self.y_test_df["Label"] = self.y_test_df["Label"].map(index_to_emotion)

    def one_hot_encode_labels(self) -> None:
        """
        Converts categorical label data into a one-hot encoded format for training, validation, and test datasets.
        """
        self.y_train_encoded = pd.get_dummies(self.y_train_df["Label"]).astype(int)
        self.y_validation_encoded = pd.get_dummies(self.y_validation_df["Label"]).astype(int)
        self.y_test_encoded = pd.get_dummies(self.y_test_df["Label"]).astype(int)

    def load_model_config(self, model_config_path: str):
        """
        Loads the model configuration from a YAML file.
        """
        with open(model_config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config

    def create_model_from_config(self, config):
        """
        Creates a model based on the configuration.
        """
        model = Sequential()
        for layer_config in config["model"]["layers"]:
            layer_type = layer_config["type"]
            # Remove 'type' from the dictionary to avoid passing it as an argument
            layer_args = {k: v for k, v in layer_config.items() if k != "type"}

            if layer_type == "Conv2D":
                model.add(Conv2D(**layer_args))
            elif layer_type == "MaxPooling2D":
                model.add(MaxPooling2D(**layer_args))
            elif layer_type == "Flatten":
                model.add(Flatten())
            elif layer_type == "Dense":
                # Special handling for 'units' if needed, e.g., if 'units' should be dynamic
                if layer_args.get("units") == "num_categories":
                    layer_args["units"] = len(self.categories)
                model.add(Dense(**layer_args))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return model

    def generate_summary(
        self,
        model_summary,
        history_output,
        test_accuracy_output,
        classification_report_output,
        accuracy_plot_path,
        confusion_matrix_path,
    ):
        """
        Process and format the outputs into a concise summary, including paths to the accuracy and confusion matrix images.
        """
        summary = f"""
        <html>
        <head><title>Model Summary - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title></head>
        <body>
        <h1>Model Summary</h1>
        <pre>{model_summary}</pre>
        <h2>Test Accuracy</h2>
        <p>{test_accuracy_output}</p>
        <h2>Accuracy Plot</h2>
        <img src="{accuracy_plot_path}" alt="Accuracy Plot">
        <h2>Classification Report</h2>
        <pre>{classification_report_output}</pre>
        <h2>History</h2>
        <pre>{history_output}</pre>
        <h2>Confusion Matrix</h2>
        <img src="{confusion_matrix_path}" alt="Confusion Matrix">
        </body>
        </html>
        """
        return summary

    def create_and_run_model(self, model_config_path: str):
        """
        Loads model configuration from a YAML file, creates the model, and runs it.
        """
        config = self.load_model_config(model_config_path)
        model = self.create_model_from_config(config)

        # Adjust the compile and train settings based on the config
        compile_config = config["model"]["compile"]
        train_config = config["model"]["train"]

        # Call the run_model function from model.py
        (
            model_summary,
            history_output,
            test_accuracy_output,
            classification_report_output,
            accuracy_plot_path,
            confusion_matrix_path,
        ) = run_model(
            model=model,
            optimizer_name=compile_config["optimizer_name"],
            learning_rate=compile_config["learning_rate"],
            epochs=train_config["epochs"],
            batch_size=train_config["batch_size"],
            patience=train_config["patience"],
            X_train=self.X_train,
            y_train_encoded=self.y_train_encoded.to_numpy(),
            X_validation=self.X_validation,
            y_validation_encoded=self.y_validation_encoded.to_numpy(),
            X_test=self.X_test,
            y_test_encoded=self.y_test_encoded.to_numpy(),
            CATEGORIES=self.categories,
        )

        # Process the outputs to generate a summary
        summary = self.generate_summary(
            model_summary,
            history_output,
            test_accuracy_output,
            classification_report_output,
            accuracy_plot_path,
            confusion_matrix_path,
        )

        # Print the summary to the terminal
        print(summary)

        # Save the summary to an HTML file
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = "./results"
        os.makedirs(results_path, exist_ok=True)
        with open(f"{results_path}/output_{current_time}.html", "w", encoding="utf-8") as f:
            f.write(summary)

    def run(self, run_config_path: str) -> None:
        """
        Executes the main functions to load and preprocess the data.
        """
        self.load_data()
        self.preprocess_data()
        self.create_and_run_model(run_config_path)


if __name__ == "__main__":
    DATA_DIR = "/home/iamtxena/sandbox/mit-ai/capstone/Facial Emotion Recognition/Facial_emotion_images"
    CATEGORIES = ["happy", "neutral", "sad", "surprise"]
    CONFIG_DIR = "/home/iamtxena/sandbox/mit-ai/capstone/Facial Emotion Recognition/model_config"
    DEFAULT_CONFIG_PATH = f"{CONFIG_DIR}/default.yaml"

    fer = FacialEmotionRecognition(DATA_DIR, CATEGORIES)

    # Check if '-all' is passed as a command-line argument
    if "-all" in sys.argv:
        # Iterate over all YAML files in the model_config directory
        for config_filename in os.listdir(CONFIG_DIR):
            if config_filename.endswith(".yaml"):
                config_path = os.path.join(CONFIG_DIR, config_filename)
                print(f"Running model with config: {config_path}")
                fer.run(config_path)
    else:
        # Use the default default.yaml if no parameter is passed
        print(f"Running model with default config: {DEFAULT_CONFIG_PATH}")
        fer.run(DEFAULT_CONFIG_PATH)

    print("FacialEmotionRecognition executed successfully.")
