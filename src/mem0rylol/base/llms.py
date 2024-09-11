from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.schema import LLMResult

class BaseLLM(ABC):
    @abstractmethod
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    @abstractmethod
    def generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        pass