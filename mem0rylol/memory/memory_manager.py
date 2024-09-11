import uuid
from typing import List, Optional, Type

from langchain.schema import Document

from mem0rylol.base.embeddings import BaseEmbeddings
from mem0rylol.base.llms import BaseLLM
from mem0rylol.base.vector_stores import BaseVectorStore
from mem0rylol.config import settings
from mem0rylol.memory.memory_types import Memory
from mem0rylol.schemas.base import BaseSchema


class MemoryManager:
    """
    @class MemoryManager
    @brief Manages the long-term and short-term memory for the AI application.
    """

    def __init__(
        self,
        table_name: str,
        schema_cls: Type[BaseSchema],
        llm: Optional[BaseLLM] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        @brief Initialize the MemoryManager.
        @param table_name The name of the table to use for the memory vector store.
        @param schema_cls The class of the schema to use for the memory vector store table.
        @param llm Optional BaseLLM instance to use for generating memories.
        @param embeddings Optional BaseEmbeddings instance to use for generating embeddings.
        @param vector_store Optional BaseVectorStore instance to use for storing memories.
        """
        self.table_name = table_name
        self.schema_cls = schema_cls
        self.llm = llm or BaseLLM()
        self.embeddings = embeddings or BaseEmbeddings()
        self.vector_store = vector_store or BaseVectorStore(self.embeddings)
        self.table = self.vector_store.create_table(self.table_name, self.schema_cls)

    def add_memory(self, memory: Memory):
        """
        @brief Add a memory to the memory vector store.
        @param memory The Memory object to add.
        """
        embedding = self.embeddings.embed_documents([memory.text])[0]
        data = self.schema_cls(id=str(uuid.uuid4()), text=memory.text, embedding=embedding)
        self.vector_store.insert_data(self.table, data)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        @brief Perform a similarity search on the memory vector store.
        @param query The query to search for.
        @param k The number of results to return.
        @return The list of Documents matching the query.
        """
        return self.vector_store.similarity_search(self.table, query, k)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """
        @brief Perform a max marginal relevance search on the memory vector store.
        @param query The query to search for.
        @param k The number of results to return.
        @param fetch_k The number of results to fetch before re-ranking.
        @return The list of Documents matching the query.
        """
        return self.vector_store.max_marginal_relevance_search(self.table, query, k, fetch_k)
