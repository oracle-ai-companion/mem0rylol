from abc import ABC
from typing import Any, Dict
from pydantic import BaseModel

class BaseSchema(BaseModel, ABC):
    """
    @class BaseSchema
    @brief Base class for defining schemas for vector store integrations.
    """
    def to_vector_store_schema(self) -> Dict[str, Any]:
        """
        @brief Convert the schema to a dictionary compatible with the vector store.
        @return A dictionary representing the schema for the vector store.
        """
        raise NotImplementedError("Subclasses must implement the to_vector_store_schema method.")