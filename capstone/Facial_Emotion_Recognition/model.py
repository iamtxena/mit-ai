"""
This module defines the structure and training process for a Facial_Emotion_Recognition model.
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
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, Nadam


class DelayedEarlyStopping(EarlyStopping):
    """Stop training when a monitored metric has stopped improving after a certain number of epochs.

    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
                   i.e., an absolute change of less than min_delta will count as no improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        verbose: Verbosity mode.
        mode: One of `{'auto', 'min', 'max'}`. In `min` mode, training will stop when the
              quantity monitored has stopped decreasing; in `max` mode it will stop when the
              quantity monitored has stopped increasing; in `auto` mode, the direction is
              automatically inferred from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity. Training will stop if the model
                  doesn't show improvement over the baseline.
        restore_best_weights: Whether to restore model weights from the epoch with the best value
                              of the monitored quantity.
        start_epoch: The epoch on which to start considering early stopping. Before this epoch,
                     early stopping will not be considered. This ensures that early stopping
                     checks only after a certain number of epochs.
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_epoch=30,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
        )
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        """Check for early stopping after the specified start epoch."""
        # Override the original `on_epoch_end` method to include `start_epoch` logic.

        # If the current epoch is less than the start epoch, skip the early stopping check
        if epoch < self.start_epoch:
            return

        # Call the parent class method to perform the regular early stopping checks after the start epoch
        super().on_epoch_end(epoch, logs)


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
    model,
    optimizer_name,
    learning_rate,
    epochs,
    batch_size,
    patience,
    train_generator=None,
    validation_generator=None,
    test_generator=None,
    X_train=None,
    y_train_encoded=None,
    X_validation=None,
    y_validation_encoded=None,
    X_test=None,
    y_test_encoded=None,
    CATEGORIES=None,
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
    elif optimizer_name.lower() == "nadam":
        optimizer = Nadam(learning_rate=learning_rate)
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
    delayed_early_stopping = DelayedEarlyStopping(
        monitor="val_loss", patience=patience, verbose=1, restore_best_weights=True, start_epoch=30
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    mc = ModelCheckpoint(
        f"{results_path}/best_model_complex_{current_time}.keras",
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    # Fitting the model
    history_buffer = io.StringIO()
    dual_output_callback = DualOutput(history_buffer)

    # Fit the model
    if train_generator and validation_generator:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[reduce_lr, mc, delayed_early_stopping, dual_output_callback],
        )
    else:
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
        # Evaluate the model
        if test_generator:
            test_loss, test_acc = model.evaluate(test_generator)
            y_true = test_generator.classes
            y_pred = model.predict(test_generator)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test_encoded, axis=1)

    test_accuracy_output = test_accuracy_buffer.getvalue()
    print(f"Test Accuracy: {test_acc} and test Loss: {test_loss} and test Accuracy: {test_accuracy_output}")

    # Generate and capture the classification report
    classification_report_buffer = io.StringIO()
    with redirect_stdout(classification_report_buffer):
        print(classification_report(y_true, y_pred, target_names=CATEGORIES))
    classification_report_output = classification_report_buffer.getvalue()

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
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
