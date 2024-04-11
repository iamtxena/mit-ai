"""
This module defines the configuration handling for the facial emotion recognition model, 
including loading configurations from YAML files and updating model parameters.
"""

import itertools
from typing import Any, Dict, List, Tuple

import yaml


class ModelConfig:
    """
    Represents the comprehensive configuration for a model run, including training parameters and model architecture.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def update_batch_config(self, learning_rate: float, optimizer: str, batch_size: int):
        """
        Updates the configuration with batch-specific parameters.

        Parameters:
        - learning_rate (float): Learning rate for the model.
        - optimizer (str): Optimizer to use.
        - batch_size (int): Batch size for training.
        """
        self.config["model"]["compile"]["learning_rate"] = learning_rate
        self.config["model"]["compile"]["optimizer"] = optimizer
        self.config["model"]["train"]["batch_size"] = batch_size

    def load_batch_config(self, batch_config_path: str) -> List[Tuple[float, str, int]]:
        """
        Loads the batch configuration from a YAML file.

        Parameters:
        - batch_config_path (str): Path to the batch configuration file.

        Returns:
        - List[Tuple[float, str, int]]: A list of tuples containing combinations of
            learning rates, optimizers, and batch sizes.
        """
        with open(batch_config_path, "r", encoding="utf-8") as file:
            batch_config = yaml.safe_load(file)

        # Create combinations of learning rates, optimizers, and batch sizes
        learning_rates = batch_config["experiments"]["learning_rates"]
        optimizers = batch_config["experiments"]["optimizers"]
        batch_sizes = batch_config["experiments"]["batch_sizes"]

        return list(itertools.product(learning_rates, optimizers, batch_sizes))


def load_config(model_config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from a YAML file.

    Parameters:
    - model_config_path (str): Path to the configuration file.

    Returns:
    - Dict[str, Any]: The loaded configuration.
    """
    with open(model_config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
