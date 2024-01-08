from dataclasses import dataclass

import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class TinyStoriesConfig:
    """TinyStories model configuration.

    Args:
    ----
        num_layers: Number of transformer layers.
        intermediate_size: Size of the intermediate layer.
        hidden_size: Size of the hidden layer.
        num_heads: Number of attention heads.

    """

    num_layers: int = 1
    intermediate_size: int = 2048
    hidden_size: int = 512
    num_heads: int = 8
    vocab_size: int = 50257
    max_position_embeddings: int = 2048


class TinyStoriesModel(LightningModule):
    """TinyStories transformer language model.

    Args:
    ----

    """

    def __init__(self, config: TinyStoriesConfig) -> None:
        super().__init__()
        self.config = GPTNeoConfig(
            num_layers=config.num_layers,
            intermediate_size=config.intermediate_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            attention_types=[[["global"], config.num_layers]],
        )
        self.model = GPTNeoForCausalLM(self.config)

    def num_params(self) -> int:
        """Get number of trainable parameters in the model.

        Returns:
        -------
            Number of trainable parameters in the model.

        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
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
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
        ----
            batch: batch of data.
            batch_idx: index of the batch.

        Returns:
        -------
            Loss tensor.

        """
        outputs = self.model(batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
