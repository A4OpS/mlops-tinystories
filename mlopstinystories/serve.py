from enum import Enum
from http import HTTPStatus

from fastapi import FastAPI
from transformers.generation import GenerationConfig

from .data import TinyStories
from .device import get_device
from .models import TinyStoriesModel


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


@app.get("/")
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/generate/{model_name}")
def read_item(model_name: str, input_text: str):
    device = get_device()

    tokenizer = TinyStories.create_tokenizer()

    model = TinyStoriesModel.load(model_name, device.torch())

    input_tokens = tokenizer(input_text, return_tensors="pt").input_ids

    generation_config = GenerationConfig(
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.5,
        do_sample=True,
    )

    output_tokens = model.generate(input_tokens.to(device.torch()), generation_config)
    output_text = tokenizer.decode(output_tokens[0].to("cpu"))

    response = {
        "message": output_text,
        "status-code": HTTPStatus.OK,
    }
    return response
