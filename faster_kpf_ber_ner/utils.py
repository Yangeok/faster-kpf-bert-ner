import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Generator

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


def chunk_text(text: str, tokenizer: PreTrainedTokenizerBase, chunk_size=512) -> Generator[list[int], None, None]:
    """텍스트를 토큰화하고 지정된 크기의 청크로 나눕니다.

    Args:
        text: 토큰화할 입력 텍스트 문자열.
        tokenizer: 토큰화에 사용할 사전 훈련된 토크나이저.
        chunk_size: 각 청크의 최대 토큰 수. 기본값은 512.

    Yields:
        List[int]: 토큰 ID의 리스트. 각 리스트는 최대 chunk_size 길이입니다.

    Example:
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> text = "This is a long text that needs to be chunked."
        >>> for chunk in chunk_text(text, tokenizer):
        ...     print(len(chunk))
        16
    """
    input_ids: torch.Tensor = tokenizer(text, return_tensors="pt")["input_ids"]
    tokens = input_ids.squeeze().tolist()
    total_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    for i in tqdm(range(0, len(tokens), chunk_size), desc="chunk", total=total_chunks):
        yield tokens[i : i + chunk_size]

class MeasureTime(ContextDecorator):
    """함수 실행 시간을 측정하는 컨텍스트 매니저 및 데코레이터.

    이 클래스는 컨텍스트 매니저로 사용하거나 함수 데코레이터로 사용할 수 있습니다.
    실행 시간을 초 단위로 출력합니다.

    Example:
        컨텍스트 매니저로 사용:
        >>> with MeasureTime():
        ...     time.sleep(1)
        elapsed time: 1.0010초

        데코레이터로 사용:
        >>> @MeasureTime()
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()
        elapsed time: 1.0015초
    """
    def __enter__(self):
        self.start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        print(f"elapsed time: {elapsed_time:.4f}초")


@dataclass
class WordList:
    """개체명 인식 결과를 저장하는 데이터 클래스.

    Attributes:
        word: 인식된 개체의 텍스트.
        label: 개체의 레이블 (예: 'PERSON', 'ORGANIZATION' 등).
        desc: 레이블에 대한 설명.
    """
    word: str
    label: str
    desc: str

