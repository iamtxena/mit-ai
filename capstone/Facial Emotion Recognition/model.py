"""
This module defines the structure and training process for a facial emotion recognition model.
It includes the setup for the model, training, evaluation, and utility functions for handling the data.
"""

import io
import os
import random
import sys
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam


class DualOutput(Callback):
    """
    Custom callback to log output to both stdout and a buffer.

    Attributes:
        buffer (io.StringIO): Buffer to which the output is also written.
    """

    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def on_epoch_end(self, epoch, logs=None):
        """
        Handles the end of an epoch event.

        Args:
            epoch (int): The index of the epoch that just ended.
            logs (dict): A dictionary of logs from the training process.
        """
        # Construct the output string
        output = f"Epoch {epoch + 1}: "
        output += ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        output += "\n"

        # Write to both sys.stdout and the buffer
        sys.stdout.write(output)
        self.buffer.write(output)


def run_model(
    model: Sequential,
    optimizer_name: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    X_train,
    y_train_encoded,
    X_validation,
    y_validation_encoded,
    X_test,
    y_test_encoded,
    CATEGORIES,
):
    """
    Clears the session, sets a random seed, compiles the model, prints a summary, performs the fit,
    plots the training and validation accuracies, checks the test accuracy, plots the confusion matrix,
    and prints the classification report. The data is passed via the project.py.
    """
    # Clearing the backend session
    tf.keras.backend.clear_session()

    # Fixing the seed for random number generators
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Setting the optimizer
    if optimizer_name.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer not supported. Use 'SGD' or 'Adam'.")

    # Compiling the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Generating the summary of the model
    model_summary_buffer = io.StringIO()
    with redirect_stdout(model_summary_buffer):
        model.summary()
    model_summary = model_summary_buffer.getvalue()

    print(model_summary)

    # Callbacks
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = "./results"
    os.makedirs(results_path, exist_ok=True)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
    mc = ModelCheckpoint(
        f"{results_path}/best_model_{current_time}.keras",
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    # Fitting the model
    history_buffer = io.StringIO()
    dual_output_callback = DualOutput(history_buffer)
    history = model.fit(
        X_train,
        y_train_encoded,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validation, y_validation_encoded),
        callbacks=[es, mc, dual_output_callback],
    )
    history_output = history_buffer.getvalue()

    # Plotting the Training and Validation Accuracies
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    accuracy_plot_path = f"{results_path}/accuracy_plot_{current_time}.png"
    plt.savefig(accuracy_plot_path)
    plt.close()

    # Checking test accuracy
    # Evaluate the model and capture the output
    test_accuracy_buffer = io.StringIO()
    with redirect_stdout(test_accuracy_buffer):
        test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
    test_accuracy_output = test_accuracy_buffer.getvalue()
    print(f"Test Accuracy: {test_acc} and test Loss: {test_loss} and test Accuracy: {test_accuracy_output}")

    # Plotting Confusion Matrix
    # Generate and capture the classification report
    classification_report_buffer = io.StringIO()
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test_encoded, axis=1)
    with redirect_stdout(classification_report_buffer):
        print(classification_report(y_true, pred, target_names=CATEGORIES))
    classification_report_output = classification_report_buffer.getvalue()

    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt=".0f", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    confusion_matrix_path = f"{results_path}/confusion_matrix_{current_time}.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Printing the classification report
    print(classification_report_output)

    # Return all captured outputs for further processing
    return (
        model_summary,
        history_output,
        test_accuracy_output,
        classification_report_output,
        accuracy_plot_path.replace("results/", ""),
        confusion_matrix_path.replace("results/", ""),
    )
