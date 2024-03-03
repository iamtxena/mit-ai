"""
This script is used to run the Facial Emotion Recognition model with a specified or default configuration.
"""

import argparse
import os
import sys

from project import FacialEmotionRecognition

if __name__ == "__main__":
    # Define the base directories
    BASE_DIR = "/home/iamtxena/sandbox/mit-ai/capstone/Facial_Emotion_Recognition"
    DATA_DIR = os.path.join(BASE_DIR, "Facial_emotion_images")
    CATEGORIES = ["happy", "neutral", "sad", "surprise"]
    CONFIG_DIR = os.path.join(BASE_DIR, "model_config")
    FINAL_MODELS_DIR = os.path.join(BASE_DIR, "final_models")
    INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")
    PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Facial Emotion Recognition model.")
    parser.add_argument(
        "-f", "--file", help="Specify the YAML configuration filename. (Do not include path)", default="default.yaml"
    )
    parser.add_argument("-p", "--predict", action="store_true", help="Enable prediction mode.")
    parser.add_argument(
        "-m", "--model", default="default.keras", help="Specify the model filename. (Do not include path)"
    )
    parser.add_argument("-d", "--directory", default=INPUT_IMAGES_DIR, help="Directory of images to predict.")
    parser.add_argument("-out", "--output", default=PREDICTIONS_DIR, help="Output directory for predictions.")
    args = parser.parse_args()

    # Construct the full path to the configuration or model file
    config_path = os.path.join(CONFIG_DIR, args.file)
    model_path = os.path.join(FINAL_MODELS_DIR, args.model)

    # Initialize the FacialEmotionRecognition instance
    fer = FacialEmotionRecognition(DATA_DIR, CATEGORIES)

    if args.predict:
        # Ensure the model file exists
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            sys.exit(1)

        # Ensure the input directory exists
        if not os.path.exists(args.directory):
            print(f"Input directory does not exist: {args.directory}")
            sys.exit(1)

        # Ensure the output directory exists or create it
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        print(f"Predicting emotions with model: {model_path} on images from: {args.directory}")
        # Call the predict method (assuming it will handle the prediction and HTML output generation)
        fer.predict(model_path, args.directory, args.output)
        print("Prediction completed. Check the output directory for results.")
    else:
        # Check if the specified configuration file exists
        if not os.path.exists(config_path):
            print(f"Configuration file does not exist: {config_path}")
            sys.exit(1)

        # Run the model with the specified or default configuration
        print(f"Running model with config: {config_path}")
        fer.run(config_path)

        print("FacialEmotionRecognition executed successfully.")
