"""
This script is used to run the Facial Emotion Recognition model with a specified or default configuration.
"""

import argparse
import itertools
import os
import sys

from model_config import ModelConfig, load_config
from project import FacialEmotionRecognition

if __name__ == "__main__":
    # Define the base directories
    BASE_DIR = "/home/iamtxena/sandbox/mit-ai/capstone/Facial_Emotion_Recognition"
    DATA_DIR = os.path.join(BASE_DIR, "Facial_emotion_images")
    CATEGORIES = ["happy", "neutral", "sad", "surprise"]
    SUBDIRS = ["train", "validation", "test"]
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
    parser.add_argument(
        "-b", "--batch", help="Specify the batch YAML configuration filename for batch mode. (Do not include path)"
    )

    args = parser.parse_args()

    # Load the main configuration
    config_path = os.path.join(CONFIG_DIR, args.file)
    main_config = load_config(config_path)
    model_config = ModelConfig(main_config)

    if args.batch:
        # Load and apply batch configuration
        batch_config_path = os.path.join(CONFIG_DIR, args.batch)
        batch_combinations = model_config.load_batch_config(batch_config_path)

        for learning_rate, optimizer, batch_size in batch_combinations:
            model_config.update_batch_config(learning_rate, optimizer, batch_size)
            # Initialize the FacialEmotionRecognition instance with the current combination
            fer = FacialEmotionRecognition(DATA_DIR, SUBDIRS, CATEGORIES, model_config)
            # Run the model
            fer.run()
            print(
                f"Run completed with learning_rate: {learning_rate}, optimizer: {optimizer}, batch_size: {batch_size}"
            )
    else:
        # Non-batch mode logic
        fer = FacialEmotionRecognition(DATA_DIR, SUBDIRS, CATEGORIES, model_config)
        if args.predict:
            # Prediction mode
            if not os.path.exists(args.model):
                print(f"Model file does not exist: {args.model}")
                sys.exit(1)
            if not os.path.exists(args.directory):
                print(f"Input directory does not exist: {args.directory}")
                sys.exit(1)
            if not os.path.exists(args.output):
                os.makedirs(args.output)

            print(f"Predicting emotions with model: {args.model} on images from: {args.directory}")
            fer.predict(args.model, args.directory, args.output)
            print("Prediction completed. Check the output directory for results.")
        else:
            # Training mode
            print(f"Running model with config: {config_path}")
            fer.run()
            print("FacialEmotionRecognition executed successfully.")
