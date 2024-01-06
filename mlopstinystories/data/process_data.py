import logging
from datetime import datetime

import torch
from data_module import TinyStories
from transformers import AutoTokenizer, PreTrainedTokenizerFast

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token

    start_time = datetime.now()
    data = TinyStories("data", tokenizer, torch.device("cpu"))
    data.prepare_data()
    end_time = datetime.now()
    print(f"Time to prepare data: {end_time - start_time}")
