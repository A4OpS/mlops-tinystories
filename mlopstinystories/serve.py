from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, Optional

from fastapi import FastAPI, Request, Response
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


@app.get("/", status_code=HTTPStatus.OK)
def read_root():
    """Health check."""
    return HTTPStatus.OK.phrase


@app.get("/generate/{model_path}", status_code=HTTPStatus.OK)
def generate(
    request: Request, response: Response, model_path: str, input: str, max_length: int = 50, temperature: float = 0.5
):
    state: AppState = request.app.state.state

    device = state.device

    tokenizer = state.tokenizer

    model = state.get_model(model_path)
    if model is None:
        response.status_code = HTTPStatus.NOT_FOUND
        return f"Model '{model_path}' not found"

    input_tokens = tokenizer(input, return_tensors="pt").input_ids

    generation_config = GenerationConfig(
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        do_sample=True,
    )

    output_tokens = model.generate(input_tokens.to(device.torch()), generation_config)
    output_text = tokenizer.decode(output_tokens[0].to("cpu"))

    return output_text
