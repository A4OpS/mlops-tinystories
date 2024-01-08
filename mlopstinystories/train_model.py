import warnings

import datasets
from datasets.dataset_dict import DatasetDict
from transformers import Trainer, TrainingArguments

from models import TinyStoriesConfig, TinyStoriesModel

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Repo card metadata block was not found. Setting CardData to empty.")
    raw_data: DatasetDict = datasets.load_dataset("roneneldan/TinyStories")  # type: ignore

    config = TinyStoriesConfig(
        num_layers=2, intermediate_size=512, hidden_size=512, num_heads=8, vocab_size=50257, max_position_embeddings=512
    )
    model = TinyStoriesModel(config)
    print(f"Number of parameters: {model.num_params()}")

    training_arguments = TrainingArguments(
        output_dir="models/test",
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=raw_data["train"],  # type: ignore
        eval_dataset=raw_data["validation"],  # type: ignore
    )

    trainer.train()
