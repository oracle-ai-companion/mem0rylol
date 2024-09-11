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
        return self.dict()

class LanceDBDocument(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Optional[dict] = None

    @property
    def page_content(self) -> str:
        return self.text

def test_lancedb_schemas():
    # Test LanceDBSchema
    schema = LanceDBSchema(id="1", text="Test text", embedding=[0.1, 0.2, 0.3])
    assert schema.id == "1"
    assert schema.text == "Test text"
    assert schema.embedding == [0.1, 0.2, 0.3]
    
    # Test to_vector_store_schema method (line 18)
    vector_store_schema = schema.to_vector_store_schema()
    assert isinstance(vector_store_schema, dict)
    assert vector_store_schema['id'] == "1"
    assert vector_store_schema['text'] == "Test text"
    assert vector_store_schema['embedding'] == [0.1, 0.2, 0.3]

    # Test LanceDBDocument
    doc = LanceDBDocument(id="2", text="Another test", embedding=[0.4, 0.5, 0.6], metadata={"key": "value"})
    assert doc.id == "2"
    assert doc.text == "Another test"
    assert doc.embedding == [0.4, 0.5, 0.6]
    assert doc.metadata == {"key": "value"}
    
    # Test page_content property (line 28)
    assert doc.page_content == "Another test"