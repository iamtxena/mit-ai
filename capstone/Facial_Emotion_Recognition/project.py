"""
This module is part of a capstone project for Facial Emotion Recognition. 
It includes the definition of the FacialEmotionRecognition class, which is responsible for loading, 
preprocessing, and managing the dataset for training, validation, and testing purposes.
"""

import base64
import os
from datetime import datetime
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import pyheif
import yaml

# Assuming model.py and prediction.py are correctly placed and accessible
from model import run_model

# Import ModelConfig from model_config.py
from model_config import ModelConfig
from PIL import Image
from prediction import predict_and_plot, predict_images
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FacialEmotionRecognition:
    """
    A class for recognizing facial emotions from images.
    """

    def __init__(self, data_dir: str, subdirs: List[str], categories: List[str], model_config: ModelConfig):
        self.data_dir = data_dir
        self.categories = categories
        self.subdirs = subdirs
        self.model_config = model_config.config  # Access the configuration dictionary directly
        self.subdirs_dict = {subdir: subdir for subdir in self.subdirs}
        self.X_train, self.y_train = [], []
        self.X_validation, self.y_validation = [], []
        self.X_test, self.y_test = [], []
        self.y_train_df, self.y_validation_df, self.y_test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.y_train_encoded, self.y_validation_encoded, self.y_test_encoded = [], [], []
        self.train_dir = os.path.join(self.data_dir, self.subdirs[0])
        self.validation_dir = os.path.join(self.data_dir, self.subdirs[1])
        self.test_dir = os.path.join(self.data_dir, self.subdirs[2])

        # Use the model_config to set up the environment
        self.batch_size = self.model_config["model"]["train"]["batch_size"]
        self.color_mode = self.model_config["model"]["data"]["color_mode"]
        self.color_layers = self.model_config["model"]["data"]["color_layers"]
        self.img_width = self.model_config["model"]["data"]["img_width"]
        self.img_height = self.model_config["model"]["data"]["img_height"]
        self.use_data_loaders = self.model_config["model"]["data"]["use_data_loaders"]

        # Initialize data generators
        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            brightness_range=(0.5, 1.5),
            shear_range=0.3,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
        )
        self.validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

    def load_data(self) -> None:
        """
        Loads images and their labels from the dataset directories.
        """

        if self.use_data_loaders:
            self.train_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                color_mode=self.color_mode,
                class_mode="categorical",
            )
            self.validation_generator = self.validation_datagen.flow_from_directory(
                self.validation_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                color_mode=self.color_mode,
                class_mode="categorical",
                shuffle=False,
            )
            self.test_generator = self.test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                color_mode=self.color_mode,
                class_mode="categorical",
                shuffle=False,
            )
        else:
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

    def create_model_from_config(self, config):
        """
        Creates a model based on the configuration.

        This method now supports Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, and ReLU layers.
        It can be easily extended to support additional layer types by adding more conditions
        and handling their specific arguments.

        Parameters:
        - config (dict): The model configuration loaded from a YAML file.

        Returns:
        - Sequential: A Keras Sequential model built according to the provided configuration.
        """
        model = Sequential()
        # With an Input layer at the beginning of the model
        model.add(Input(shape=(self.img_width, self.img_height, self.color_layers)))

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
            elif layer_type == "Dropout":
                model.add(Dropout(**layer_args))
            elif layer_type == "BatchNormalization":
                model.add(BatchNormalization())
            elif layer_type == "LeakyReLU":
                model.add(LeakyReLU(**layer_args))
            elif layer_type == "ReLU":
                model.add(ReLU())
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
        predicted_image_filename,
    ):
        """
        Process and format the outputs into a concise summary, including paths to the accuracy and confusion matrix images.
        Now also includes model configuration parameters before the model summary.
        """
        # Convert the model configuration to a formatted string
        config_str = yaml.dump(self.model_config)

        summary = f"""
        <html>
        <head><title>Model Summary - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title></head>
        <body>
        <h1>Model Configuration</h1>
        <pre>{config_str}</pre>
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
        <h2>Predicted Images</h2>
        <img src="./{predicted_image_filename}" alt="Predicted Images">
        </body>
        </html>
        """
        return summary

    def create_and_run_model(self):
        """
        Creates the model and runs it based on the configuration provided during class initialization.
        """
        # Access configuration directly from self.model_config
        compile_config = self.model_config["model"]["compile"]
        train_config = self.model_config["model"]["train"]

        # Create the model
        model = self.create_model_from_config(self.model_config)

        if self.use_data_loaders:
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
                train_generator=self.train_generator,
                validation_generator=self.validation_generator,
                test_generator=self.test_generator,
                CATEGORIES=self.categories,
            )
            predicted_image_filename = predict_and_plot(model, self.test_generator, None, self.categories)
        else:

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

            predicted_image_filename = predict_and_plot(model, self.X_test, self.y_test_encoded, self.categories)

        # Process the outputs to generate a summary
        summary = self.generate_summary(
            model_summary,
            history_output,
            test_accuracy_output,
            classification_report_output,
            accuracy_plot_path,
            confusion_matrix_path,
            predicted_image_filename,
        )

        # Print the summary to the terminal
        print(summary)

        # Save the summary to an HTML file
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = "./results"
        os.makedirs(results_path, exist_ok=True)
        with open(f"{results_path}/output_{current_time}.html", "w", encoding="utf-8") as f:
            f.write(summary)

    def run(self) -> None:
        """
        Executes the main functions to load and preprocess the data, and create and run the model.
        """
        self.load_data()
        self.preprocess_data()
        self.create_and_run_model()

    def predict(self, model_path: str, images_dir: str, output_dir: str) -> None:
        """
        Predicts emotions for images in a specified directory using a trained model and generates an HTML output.

        Parameters:
        - model_path (str): Path to the trained model file.
        - images_dir (str): Directory containing images to predict.
        - output_dir (str): Directory to save the HTML output.
        """
        # Load the model
        model = load_model(model_path)

        # Load and preprocess images
        images = []
        for img_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_name)
            try:
                # Check the file extension and process accordingly
                if img_path.lower().endswith(".heic"):
                    heif_file = pyheif.read(img_path)
                    img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    ).convert(
                        "L"
                    )  # Convert to grayscale
                else:
                    img = Image.open(img_path).convert("L")  # Convert to grayscale

                img = img.resize((48, 48))  # Resize to 48x48
                img_array = np.array(img) / 255.0  # Normalize pixels
                images.append(img_array)
            except (IOError, ValueError) as e:
                print(f"Error processing image {img_name}: {e}")

        images = np.expand_dims(np.array(images), axis=-1)  # Add channel dimension

        if self.use_data_loaders:
            test_generator = self.test_datagen.flow_from_directory(
                images_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=1,
                color_mode=self.color_mode,
                class_mode="categorical",
                shuffle=False,
            )
            predicted_emotions = predict_images(model, test_generator, self.categories)
        else:
            # Predict emotions
            predicted_emotions = predict_images(model, images, self.categories)

        # Generate HTML output
        self.generate_html_output(predicted_emotions, images, output_dir, self.model_config)

    def generate_html_output(self, predicted_emotions, images, output_dir, model_config):
        """
        Generates an HTML file displaying images with their predicted emotions and includes the model configuration.

        Parameters:
        - predicted_emotions (List[str]): List of predicted emotions.
        - images (np.ndarray): Array of preprocessed images.
        - output_dir (str): Directory to save the HTML output.
        - model_config (ModelConfig): The configuration object with all the settings used for the model.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        html_path = os.path.join(output_dir, f"prediction_{timestamp}.html")

        # Convert the model configuration to a formatted string
        config_str = yaml.dump(model_config.config)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><title>Predictions</title></head><body>")
            f.write("<h1>Predictions</h1>")
            f.write("<div style='display: flex; flex-wrap: wrap;'>")

            for emotion, img in zip(predicted_emotions, images):
                img_encoded = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
                buffered = BytesIO()
                img_encoded.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                f.write(
                    f"<div style='margin: 10px;'><img src='data:image/png;base64,{img_str}' title='Predicted: {emotion}' style='width:200px; height:auto;'/><p>Predicted: {emotion}</p></div>"
                )

            f.write("</div>")
            f.write("<h2>Model Configuration</h2>")
            f.write(f"<pre>{config_str}</pre>")
            f.write("</body></html>")

        print(f"Predictions saved to {html_path}")
