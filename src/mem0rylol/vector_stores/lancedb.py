from langchain.vectorstores import LanceDB
from langchain.docstore.document import Document
from lancedb.pydantic import LanceModel, Vector
from mem0rylol.base.vector_stores import BaseVectorStore

class MemoryVectorStore(BaseVectorStore):
    def __init__(self, embedding_function, connection=None, local_path=None):
        self.embedding_function = embedding_function
        
        if connection is not None:
            self.connection = connection
        elif local_path is not None:
            os.makedirs(local_path, exist_ok=True)
            self.connection = lancedb.connect(local_path)
        else:
            raise ValueError("Either connection or local_path must be provided.")
        
    def create_table(self, table_name, schema):
        return self.connection.create_table(table_name, schema=schema)
    
    def insert_data(self, table, data):
        table.add(data)
        
    def similarity_search(self, table, query, k=4):
        query_embedding = self.embedding_function.embed_query(query)
        docs = table.search(query_embedding).limit(k).to_pydantic()
        return [Document(page_content=doc.text, metadata=doc.dict()) for doc in docs]
        
    def max_marginal_relevance_search(self, table, query, k=4, fetch_k=20):
        query_embedding = self.embedding_function.embed_query(query)
        docs = table.search(query_embedding).limit(fetch_k).to_pydantic() 
        docs = LanceDB.max_marginal_relevance_search(query_embedding, docs, k=k)
        return [Document(page_content=doc.text, metadata=doc.dict()) for doc in docs]
