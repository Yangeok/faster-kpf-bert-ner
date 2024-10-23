import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Generator

import torch
from transformers import PreTrainedTokenizerBase


def chunk_text(text: str, tokenizer: PreTrainedTokenizerBase, chunk_size=512) -> Generator[list[int], None, None]:
    input_ids: torch.Tensor = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
    tokens = input_ids.squeeze().tolist()
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i : i + chunk_size]

class MeasureTime(ContextDecorator):
    def __enter__(self):
        self.start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        print(f"elapsed time: {elapsed_time:.4f}초")


@dataclass
class WordList:
    word: str
    label: str
    desc: str

