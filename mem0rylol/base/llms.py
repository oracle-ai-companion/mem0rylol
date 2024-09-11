from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.schema import LLMResult


class BaseLLM(ABC):
    @abstractmethod
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError("Subclasses must implement __call__ method.")

    @abstractmethod
    def generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        raise NotImplementedError("Subclasses must implement generate method.")
