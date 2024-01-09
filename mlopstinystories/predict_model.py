import torch
from device import get_device
from transformers.generation import GenerationConfig

from data import TinyStories
from models import TinyStoriesModel


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
    ----
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns:
    -------
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


if __name__ == "__main__":
    device = get_device()

    data = TinyStories("data", device.torch())
    tokenizer = data.tokenizer

    model = TinyStoriesModel.load("model1", device.torch())

    input_text = "Once upon a time,"
    input_tokens = tokenizer(input_text, return_tensors="pt").input_ids
    print("Input tokens: ", input_tokens)

    for _ in range(10):
        generation_config = GenerationConfig(max_length=50, pad_token_id=50000)
        output_tokens = model.generate(input_tokens.to(device.torch()), generation_config)
        print("Output tokens: ", output_tokens)
        output_text = tokenizer.decode(output_tokens[0].to("cpu"))
        print("Output text: ", output_text)
