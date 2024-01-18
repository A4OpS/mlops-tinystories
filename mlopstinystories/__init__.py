"""Replication of the Tiny Stories paper."""
from .train_model import TrainModelConfig
from .train_model import main as train_model

__all__ = ["train_model", "TrainModelConfig"]
