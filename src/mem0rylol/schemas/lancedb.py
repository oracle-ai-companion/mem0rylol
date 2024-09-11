from lancedb.pydantic import LanceModel, Vector
from src.schemas.base import BaseSchema

class LanceDBSchema(BaseSchema, LanceModel):
    """
    @class LanceDBSchema
    @brief Defines the schema for the LanceDB vector store integration.
    """
    embedding: Vector

    def to_vector_store_schema(self):
        return self.dict()