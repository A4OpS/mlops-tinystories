from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, Optional

from fastapi import FastAPI, Request
from transformers import PreTrainedTokenizerFast
from transformers.generation import GenerationConfig

from .data import TinyStories
from .device import Device, get_device
from .models import ModelNotFoundError, TinyStoriesModel


class AppState:
    _device: Device
    _tokenizer: PreTrainedTokenizerFast
    _models: Dict[str, TinyStoriesModel]

    def __init__(
        self,
        device: Device,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self._device = device
        self._tokenizer = tokenizer
        self._models = {}

    @property
    def device(self) -> Device:
        return self._device

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    def get_model(self, model_path: str) -> Optional[TinyStoriesModel]:
        if model_path not in self._models:
            try:
                model = TinyStoriesModel.load(model_path, self.device.torch())
                self._models[model_path] = model
            except ModelNotFoundError:
                return None

        return self._models[model_path]


@asynccontextmanager
async def initialize(app: FastAPI):
    device = get_device()
    tokenizer = TinyStories.create_tokenizer()
    app.state.state = AppState(device, tokenizer)
    yield


app = FastAPI(lifespan=initialize)


@app.get("/")
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/generate/{model_path}")
def generate(model_path: str, input_text: str, request: Request):
    state: AppState = request.app.state.state

    device = state.device

    tokenizer = state.tokenizer

    model = state.get_model(model_path)
    if model is None:
        response = {
            "message": f"Model '{model_path}' not found",
            "status-code": HTTPStatus.NOT_FOUND,
        }
        return response

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
