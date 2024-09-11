from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document


class BaseVectorStore(ABC):
    @abstractmethod
    def create_table(self, table_name: str, schema: dict):
        raise NotImplementedError("Subclasses must implement create_table method.")

    @abstractmethod
    def insert_data(self, table, data):
        raise NotImplementedError("Subclasses must implement insert_data method.")

    @abstractmethod
    def similarity_search(self, table, query: str, k: int = 4) -> List[Document]:
        raise NotImplementedError("Subclasses must implement similarity_search method.")

    @abstractmethod
    def max_marginal_relevance_search(
        self, table, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        raise NotImplementedError("Subclasses must implement max_marginal_relevance_search method.")

    @abstractmethod
    def delete(self, table, ids: List[str]):
        raise NotImplementedError("Subclasses must implement delete method.")
