import os

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from mlopstinystories import TrainModelConfig, train_model
from mlopstinystories.data import TinyStories, fetch_raw_data
from mlopstinystories.models import ModelNotFoundError, TinyStoriesModel, TinyStoriesModelConfig


def test_model():

    config = TinyStoriesModelConfig()

    # check attrubutes of config
    assert hasattr(config, "num_layers")
    assert hasattr(config, "intermediate_size")
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_heads")
    assert hasattr(config, "num_heads")
    assert hasattr(config, "vocab_size")
    assert hasattr(config, "max_position_embeddings")

    #  check model with default config
    try:
        model = TinyStoriesModel.initialize(config, torch.device("cpu"))
    except ModelNotFoundError as err:
        raise AssertionError("Standard config model not found") from err

    # Standard input
    input_ids = TinyStories.create_tokenizer()(
        "The quick brown fox jumps over the lazy dog", return_tensors="pt"
        ).input_ids

    # check model output
    output = model(input_ids)
    assert isinstance(output,CausalLMOutputWithPast)
    # Shape logits: batch size x sequence length (x vocab size)
    assert output.logits.shape[:2] == torch.Size([1,9]), "logits shape incorrect"
    assert output.logits.requires_grad is True, "logits should be trainable (requires grad)"
    assert output.past_key_values is not None, "past_key_values should not be None"

def test_train():
    # fetch data
    fetch_raw_data()

    # train model configs
    config = TrainModelConfig()
    config.max_steps = 2
    config.batch_size = 1
    try:
        train_model(os.getcwd(),config)
    except Exception as err:
        raise AssertionError("Training failed") from err
