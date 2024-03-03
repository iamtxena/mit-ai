"""
This script is used to run the Facial Emotion Recognition model with a specified or default configuration.
"""

import argparse
import os
import sys

from project import FacialEmotionRecognition

if __name__ == "__main__":
    # Define the base directories
    DATA_DIR = "/home/iamtxena/sandbox/mit-ai/capstone/Facial_Emotion_Recognition/Facial_emotion_images"
    CATEGORIES = ["happy", "neutral", "sad", "surprise"]
    CONFIG_DIR = "/home/iamtxena/sandbox/mit-ai/capstone/Facial_Emotion_Recognition/model_config"

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Facial Emotion Recognition model.")
    parser.add_argument(
        "-f", "--file", help="Specify the YAML configuration filename. (Do not include path)", default="default.yaml"
    )
    args = parser.parse_args()

    # Construct the full path to the configuration file
    config_filename = args.file
    config_path = os.path.join(CONFIG_DIR, config_filename)

    # Initialize the FacialEmotionRecognition instance
    fer = FacialEmotionRecognition(DATA_DIR, CATEGORIES)

    # Check if the specified configuration file exists
    if not os.path.exists(config_path):
        print(f"Configuration file does not exist: {config_path}")
        sys.exit(1)

    # Run the model with the specified or default configuration
    print(f"Running model with config: {config_path}")
    fer.run(config_path)

    print("FacialEmotionRecognition executed successfully.")
