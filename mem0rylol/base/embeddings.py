from abc import ABC, abstractmethod

class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts):
        raise NotImplementedError("Subclasses must implement embed_documents method.")

    @abstractmethod
    def embed_query(self, text):
        raise NotImplementedError("Subclasses must implement embed_query method.")