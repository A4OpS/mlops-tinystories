import torch
from transformers.generation import GenerationConfig

from data import TinyStories
from device import get_device
from models import TinyStoriesModel


def main():
    device = get_device()

    tokenizer = TinyStories.create_tokenizer()

    model = TinyStoriesModel.load("model1", device.torch())

    input_text = "Once upon a time, there was a girl named Alice. She was very silly and loved to"
    input_tokens = tokenizer(input_text, return_tensors="pt").input_ids
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

    for _ in range(10):
        generation_config = GenerationConfig(max_length=50, pad_token_id=50000, temperature=0.5, do_sample=True)
        output_tokens = model.generate(input_tokens.to(device.torch()), generation_config)
        print("Output tokens: ", output_tokens)
        output_text = tokenizer.decode(output_tokens[0].to("cpu"))
        print("Output text: ", output_text)


if __name__ == "__main__":
    main()
