from lancedb.pydantic import LanceModel, Vector
from mem0rylol.schemas.base import BaseSchema
from pydantic import ConfigDict, BaseModel, Field
from typing import List, Optional

class LanceDBSchema(BaseSchema, LanceModel):
    """
    @class LanceDBSchema
    @brief Defines the schema for the LanceDB vector store integration.
    """
    id: str = Field(description="Unique identifier for the document")
    text: str = Field(description="The text content of the document")
    embedding: List[float] = Field(description="The vector embedding of the document")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_vector_store_schema(self):
        return self.model_dump()

class LanceDBDocument(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Optional[dict] = None

    @property
    def page_content(self) -> str:
        return self.text

    @classmethod
    def from_vector_store_schema(cls, data: dict) -> 'LanceDBDocument':
        return cls(
            id=data['id'],
            text=data['text'],
            embedding=data['embedding'],
            metadata=data.get('metadata')
        )