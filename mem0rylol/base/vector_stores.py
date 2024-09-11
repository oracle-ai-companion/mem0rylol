from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class BaseVectorStore(ABC):
    @abstractmethod
    def create_table(self, table_name: str, schema: dict):
        pass

    @abstractmethod
    def insert_data(self, table, data):
        pass

    @abstractmethod
    def similarity_search(self, table, query: str, k: int = 4) -> List[Document]:
        pass

    @abstractmethod
    def max_marginal_relevance_search(self, table, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
        pass