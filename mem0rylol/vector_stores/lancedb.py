from langchain_community.vectorstores import LanceDB
from langchain.docstore.document import Document
from lancedb.pydantic import LanceModel, Vector
from mem0rylol.base.vector_stores import BaseVectorStore
from mem0rylol.config import settings
import os
import lancedb

class MemoryVectorStore(BaseVectorStore):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        
        connection_string = settings.LANCEDB_CONNECTION_STRING
        if connection_string.startswith(('http://', 'https://')):
            self.connection = lancedb.connect(connection_string)
        else:
            os.makedirs(connection_string, exist_ok=True)
            self.connection = lancedb.connect(connection_string)

    def create_table(self, table_name, schema):
        return self.connection.create_table(table_name, schema=schema.model_json_schema())

    def insert_data(self, table, data):
        table.add([data.model_dump()])
        
    def similarity_search(self, table, query, k=4):
        query_embedding = self.embedding_function.embed_query(query)
        docs = table.search(query_embedding).limit(k).to_pydantic()
        return [Document(page_content=doc.text, metadata={"id": doc.id}) for doc in docs]
        
    def max_marginal_relevance_search(self, table, query, k=4, fetch_k=20):
        query_embedding = self.embedding_function.embed_query(query)
        docs = table.search(query_embedding).limit(fetch_k).to_pydantic()
        # Implement MMR logic here
        return [Document(page_content=doc.text, metadata={"id": doc.id}) for doc in docs[:k]]
