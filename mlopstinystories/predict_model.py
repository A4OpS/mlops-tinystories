from dataclasses import dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore
from transformers.generation import GenerationConfig

from data import TinyStories
from device import get_device
from models import TinyStoriesModel


@dataclass
class PredictConfig:
    model: str
    input_text: str
    temperature: float
    num_samples: int
    max_sample_length: int


cs = ConfigStore.instance()

cs.store(name="predict_config", node=PredictConfig)


@hydra.main(config_path="../conf/predict", version_base="1.3")
def main(config: PredictConfig) -> None:
    device = get_device()

    tokenizer = TinyStories.create_tokenizer()

    model = TinyStoriesModel.load(config.model, device.torch())

    input_tokens = tokenizer(config.input_text, return_tensors="pt").input_ids
    print("Input tokens: ", input_tokens)

    output = model(input_tokens.to(device.torch()))
    logits = output.logits
    print("Logits: ", logits.shape)
    gen_logits = logits[0, -1, :]
    top_tokens = torch.topk(gen_logits, 10).indices
    top_probs = torch.topk(gen_logits, 10).values
    print(top_tokens)
    print(top_probs)
    print("Top tokens: ", top_tokens.shape)
    print([tokenizer.decode(token) for token in top_tokens])

    for _ in range(config.num_samples):
        generation_config = GenerationConfig(
            max_length=config.max_sample_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=config.temperature,
            do_sample=True,
        )
        output_tokens = model.generate(input_tokens.to(device.torch()), generation_config)
        output_text = tokenizer.decode(output_tokens[0].to("cpu"))
        print("Output text: ", output_text)


if __name__ == "__main__":
    main()
