import os
import typing
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import LongTensor
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput

MODELS_DIR = "models"


@dataclass
class TinyStoriesModelConfig:
    """TinyStories model configuration.

    Args:
    ----
        num_layers (int): Number of transformer layers.
        intermediate_size (int): Size of the intermediate layer.
        hidden_size (int): Size of the hidden layer.
        num_heads (int): Number of attention heads.
        vocab_size (int): Size of vocabulary.
        max_position_embeddings (int): The maximum sequence length that this model might ever be used with.
    """

    num_layers: int = 1
    intermediate_size: int = 2048
    hidden_size: int = 512
    num_heads: int = 8
    vocab_size: int = 50257
    max_position_embeddings: int = 2048


class ModelNotFoundError(Exception):
    """Model not found exception."""

    _model_path: str

    def __init__(self, model_path: str) -> None:
        super().__init__(f"Model {model_path} not found")
        self._model_path = model_path

    @property
    def model_path(self) -> str:
        """Get model name.

        Returns:
        -------
            Model name.

        """
        return self._model_path


class TinyStoriesModel(LightningModule):
    _model: GPTNeoForCausalLM

    """TinyStories transformer language model.

    Args:
    ----

    """

    def __init__(self, model: GPTNeoForCausalLM) -> None:
        super().__init__()
        self._model = model

    @staticmethod
    def initialize(config: TinyStoriesModelConfig, device: torch.device) -> "TinyStoriesModel":
        """Initialize the model with the given configuration."""
        model_config = GPTNeoConfig(
            num_layers=config.num_layers,
            intermediate_size=config.intermediate_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            attention_types=[[["global"], config.num_layers]],
        )
        model = GPTNeoForCausalLM(model_config)
        model = model.to(device=device)  # type: ignore
        return TinyStoriesModel(model)

    @staticmethod
    def load(path: str, device: torch.device) -> "TinyStoriesModel":
        """Load the model from the given path within the models directory."""

        whole_path = os.path.join(MODELS_DIR, path)
        if not os.path.exists(MODELS_DIR):
            raise Exception("Models directory does not exist.")
        if not os.path.exists(whole_path):
            raise ModelNotFoundError(path)
        model = typing.cast(GPTNeoForCausalLM, GPTNeoForCausalLM.from_pretrained(whole_path))
        model = model.to(device=device)  # type: ignore
        return TinyStoriesModel(model)

    def save(self, path: str) -> None:
        """Save the model to the given path within the models directory."""

        whole_path = os.path.join(MODELS_DIR, path)
        path_parent = os.path.dirname(whole_path)
        if not os.path.exists(path_parent):
            os.makedirs(path_parent)
        self._model.save_pretrained(whole_path)

    def device(self) -> torch.device:
        """Get device of the model.

        Returns:
        -------
            Device of the model.

        """
        return self._model.device

    def generate(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> ModelOutput | LongTensor:
        """Generate text from the given inputs.

        Args:
        ----
            inputs: input tensor of shape [N, in_features]
            generation_config: generation configuration

        Returns:
        -------
            Output tensor of shape [N, out_features]

        """
        return self._model.generate(inputs, generation_config)

    def num_params(self) -> int:
        """Get number of trainable parameters in the model.

        Returns:
        -------
            Number of trainable parameters in the model.

        """
        model_parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        return sum([int(np.prod(p.size())) for p in model_parameters])

    def forward(self, x: torch.Tensor) -> CausalLMOutputWithCrossAttentions:
        """Forward pass of the model.

        Args:
        ----
            x: input tensor expected to be of shape [N,in_features]

        Returns:
        -------
            Output tensor with shape [N,out_features]

        """
        return self._model(x)

    def training_step(self, batch_list: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
        ----
            batch: batch of data.
            batch_idx: index of the batch.

        Returns:
        -------
            Loss tensor.

        """
        [batch] = batch_list
        outputs = self._model(batch[:-1], labels=batch[1:])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch_list: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
        ----
            batch: batch of data.
            batch_idx: index of the batch.

        Returns:
        -------
            Loss tensor.

        """
        [batch] = batch_list
        outputs = self._model(batch[:-1], labels=batch[1:])
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=1e-4)
