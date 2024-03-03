# README.md for Facial Emotion Recognition Project

## Overview

The Facial Emotion Recognition (FER) project is a machine learning application designed to identify human emotions from facial expressions in images. The system uses a pre-trained model to predict emotions such as happiness, neutrality, sadness, and surprise.

## Project Structure

The project is structured as follows:

- `run.py`: The main script to run the FER model with various configurations.
- `project.py`: Contains the [FacialEmotionRecognition](vscode-remote://ssh-remote%2Bchengdu.home/home/iamtxena/sandbox/mit-ai/capstone/Facial_Emotion_Recognition/run.py#9%2C21-9%2C21) class with methods for running predictions and generating output.
- `model_config/`: Directory containing YAML configuration files for the model.
- [final_models/](vscode-remote://ssh-remote%2Bchengdu.home/home/iamtxena/sandbox/mit-ai#925%2C99-925%2C99): Directory where the trained model files are stored.
- `input_images/`: Directory containing images on which predictions are to be made.
- `predictions/`: Directory where the prediction outputs (HTML files) are saved.
- `Facial_emotion_images/`: Directory with datasets used for training and testing the model.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- Required Python packages: [tensorflow](vscode-remote://ssh-remote%2Bchengdu.home/home/iamtxena/sandbox/mit-ai#921%2C31-921%2C31), `numpy`, `PIL`, `pyheif-pillow-opener` (for HEIC image support)

## Running the Project

To run the project, use the `run.py` script with the following options:

- `-f` or `--file`: Specify the YAML configuration filename (without the path).
- `-p` or `--predict`: Enable prediction mode to make predictions on images.
- `-m` or `--model`: Specify the model filename (without the path).
- `-d` or `--directory`: Directory containing images to predict.
- `-out` or `--output`: Output directory for predictions.

### Prediction Mode

To make predictions on images:

```bash
python run.py -p -m <model_name>.keras -d <path_to_image_directory> -out <path_to_output_directory>
```

Replace `<model_name>` with the name of your model file, `<path_to_image_directory>` with the path to the directory containing input images, and `<path_to_output_directory>` with the path where you want the predictions to be saved.

### Configuration Mode

To run the model with a specific configuration:

```bash
python run.py -f <config_file>.yaml
```

Replace `<config_file>` with the name of your configuration file.

## Inputs and Outputs

- Input images should be placed in the `input_images/` directory or a directory specified by the `-d` option.
- The model file should be placed in the `final_models/` directory or specified with the `-m` option.
- The output will be saved in the `predictions/` directory or a directory specified by the `-out` option.

## Example Usage

Running predictions with the default model and images:

```bash
python run.py -p
```

Running the model with a custom configuration file:

```bash
python run.py -f custom_config.yaml
```

## Output

After running predictions, check the specified output directory for an HTML file containing the predictions. Each image will be displayed with the predicted emotion label.

For any issues or further instructions, refer to the documentation within each script and module docstrings.
