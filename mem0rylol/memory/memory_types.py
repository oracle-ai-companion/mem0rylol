from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """
    @class Memory
    @brief Represents a memory in the vector store.
    """

    id: Optional[str] = Field(default_factory=lambda: str(datetime.now().timestamp()))
    text: str
    metadata: Optional[dict] = None
    embedding: Optional[list] = None
