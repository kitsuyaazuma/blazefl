"""
Core module of BlazeFL framework.

This module imports and defines the core components of the BlazeFL framework,
including client trainers, model selectors, partitioned datasets, and server handlers.
"""

from blazefl.core.client_trainer import (
    BaseClientTrainer,
    ProcessPoolClientTrainer,
    ThreadPoolClientTrainer,
)
from blazefl.core.model_selector import ModelSelector
from blazefl.core.partitioned_dataset import PartitionedDataset
from blazefl.core.server_handler import BaseServerHandler

__all__ = [
    "BaseClientTrainer",
    "ProcessPoolClientTrainer",
    "ThreadPoolClientTrainer",
    "ModelSelector",
    "PartitionedDataset",
    "BaseServerHandler",
]
