import torch
from typing import Protocol

class ModelSelector(Protocol):
    def select_model(self, model_name: str) -> torch.nn.Module: ...
