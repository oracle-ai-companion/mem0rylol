import pytest
from unittest.mock import Mock, patch, MagicMock
from mem0rylol.memory.memory_manager import MemoryManager
from mem0rylol.schemas.lancedb import LanceDBSchema, LanceDBDocument
from mem0rylol.embeddings.google_genai import GoogleGenAIEmbeddings
from mem0rylol.llms.cerebras import CerebrasLLM
from mem0rylol.vector_stores.lancedb import MemoryVectorStore
from mem0rylol.memory.memory_types import Memory
from mem0rylol.config import settings
from langchain.schema import Document, LLMResult, Generation
import lancedb
from mem0rylol.schemas.base import BaseSchema
from mem0rylol.base.embeddings import BaseEmbeddings
from mem0rylol.base.llms import BaseLLM
from mem0rylol.base.vector_stores import BaseVectorStore
from typing import Optional, List

@pytest.fixture
def mock_embeddings():
    with patch('mem0rylol.embeddings.google_genai.GoogleGenerativeAIEmbeddings') as mock:
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_llm():
    with patch('mem0rylol.llms.cerebras.ChatCerebras') as mock:
        mock_instance = Mock()
        mock_instance.return_value = "Mocked response" 
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_vector_store():
    with patch('mem0rylol.vector_stores.lancedb.MemoryVectorStore') as mock_vector_store:
        mock_instance = MagicMock()
        mock_vector_store.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def memory_manager(mock_embeddings, mock_llm, mock_vector_store):
    embeddings = mock_embeddings
    llm = mock_llm
    vector_store = mock_vector_store
    return MemoryManager(table_name="test_memories", schema_cls=LanceDBSchema, llm=llm, embeddings=embeddings, vector_store=vector_store)

def test_add_memory(memory_manager, mock_vector_store):
    memory = Memory(text="The capital of France is Paris.")
    memory_manager.add_memory(memory)
    
    mock_vector_store.insert_data.assert_called_once()

def test_similarity_search(memory_manager, mock_vector_store):
    mock_docs = [Document(page_content="Paris is the capital of France", metadata={"id": "1"})]
    mock_vector_store.similarity_search.return_value = mock_docs
    
    result = memory_manager.similarity_search("What is the capital of France?")
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Paris is the capital of France"

def test_max_marginal_relevance_search(memory_manager, mock_vector_store):
    mock_docs = [
        Document(page_content=f"Programming language {i}", metadata={"id": str(i)})
        for i in range(5)
    ]
    mock_vector_store.max_marginal_relevance_search.return_value = mock_docs[:3]
    
    result = memory_manager.max_marginal_relevance_search("What are some popular programming languages?", k=3, fetch_k=5)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(doc, Document) for doc in result)
    assert all("Programming language" in doc.page_content for doc in result)

def test_config_validation():
    assert settings.LANCEDB_CONNECTION_STRING is not None
    assert isinstance(settings.LANCEDB_CONNECTION_STRING, str)

def test_google_genai_embeddings():
    with patch('mem0rylol.embeddings.google_genai.GoogleGenerativeAIEmbeddings') as mock_embeddings:
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_instance

        embeddings = GoogleGenAIEmbeddings()
        
        result = embeddings.embed_documents(["Test document"])
        assert result == [[0.1, 0.2, 0.3]]
        
        result = embeddings.embed_query("Test query")
        assert result == [0.1, 0.2, 0.3]

def test_cerebras_llm():
    with patch('mem0rylol.llms.cerebras.ChatCerebras') as mock_cerebras:
        mock_instance = Mock()
        mock_instance.return_value = "Mocked response"
        mock_instance.generate.return_value = LLMResult(generations=[
            [Generation(text="Mocked generation 1")],
            [Generation(text="Mocked generation 2")]
        ])
        mock_cerebras.return_value = mock_instance

        llm = CerebrasLLM("test-model")
        
        result = llm("Test prompt")
        assert result == "Mocked response"
        
        result = llm.generate(["Test prompt 1", "Test prompt 2"])
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 2
        assert result.generations[0][0].text == "Mocked generation 1"
        assert result.generations[1][0].text == "Mocked generation 2"

def test_memory_vector_store():
    with patch('mem0rylol.vector_stores.lancedb.lancedb') as mock_lancedb:
        mock_connection = Mock()
        mock_table = Mock()
        mock_connection.create_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_connection

        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        vector_store = MemoryVectorStore(mock_embeddings)
        
        # Test create_table
        table = vector_store.create_table("test_table", LanceDBSchema)
        assert table == mock_table
        
        # Test insert_data
        data = LanceDBSchema(id="1", text="Test", embedding=[0.1, 0.2, 0.3])
        vector_store.insert_data(mock_table, data)
        mock_table.add.assert_called_once()
        
        # Test similarity_search
        mock_table.search.return_value.limit.return_value.to_pydantic.return_value = [
            LanceDBSchema(id="1", text="Test result", embedding=[0.1, 0.2, 0.3])
        ]
        results = vector_store.similarity_search(mock_table, "Test query")
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content == "Test result"
        assert results[0].metadata == {"id": "1"}
        
        # Test max_marginal_relevance_search
        results = vector_store.max_marginal_relevance_search(mock_table, "Test query")
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content == "Test result"
        assert results[0].metadata == {"id": "1"}

def test_base_schema():
    class TestSchema(BaseSchema):
        def to_vector_store_schema(self):
            return {"test": "data"}

    schema = TestSchema()
    result = schema.to_vector_store_schema()
    assert result == {"test": "data"}

    with pytest.raises(NotImplementedError):
        BaseSchema().to_vector_store_schema()

def test_base_embeddings():
    class TestEmbeddings(BaseEmbeddings):
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    embeddings = TestEmbeddings()
    assert embeddings.embed_documents(["test1", "test2"]) == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert embeddings.embed_query("test") == [0.1, 0.2, 0.3]

    with pytest.raises(TypeError):
        BaseEmbeddings()

def test_base_llm():
    class TestLLM(BaseLLM):
        def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            return "Test response"

        def generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
            return LLMResult(generations=[[Generation(text="Test generation")] for _ in prompts])

    llm = TestLLM()
    assert llm("Test prompt") == "Test response"
    result = llm.generate(["Prompt 1", "Prompt 2"])
    assert isinstance(result, LLMResult)
    assert len(result.generations) == 2
    assert result.generations[0][0].text == "Test generation"

    with pytest.raises(TypeError):
        BaseLLM()

def test_base_vector_store():
    class TestVectorStore(BaseVectorStore):
        def create_table(self, table_name: str, schema: dict):
            return "Test table"

        def insert_data(self, table, data):
            pass

        def similarity_search(self, table, query: str, k: int = 4) -> List[Document]:
            return [Document(page_content="Test document")]

        def max_marginal_relevance_search(self, table, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
            return [Document(page_content="Test document")]

    vector_store = TestVectorStore()
    assert vector_store.create_table("test", {}) == "Test table"
    vector_store.insert_data("test_table", "test_data")
    assert len(vector_store.similarity_search("test_table", "query")) == 1
    assert len(vector_store.max_marginal_relevance_search("test_table", "query")) == 1

    with pytest.raises(TypeError):
        BaseVectorStore()

# Add more tests as needed