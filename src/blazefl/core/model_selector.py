from typing import Protocol

import torch


class ModelSelector(Protocol):
    """
    Abstract base class for selecting models in federated learning.

    This class defines the interface for selecting and retrieving models
    based on a given model name.

    Raises:
        NotImplementedError: If the method is not implemented in a subclass.
    """

    def select_model(self, model_name: str) -> torch.nn.Module:
        """
        Select and return a model instance by its name.

        Args:
            model_name (str): The name of the model to select.

        Returns:
            torch.nn.Module: An instance of the selected model.
        """
        ...
