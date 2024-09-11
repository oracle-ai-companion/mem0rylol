from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import lancedb
import pytest
from langchain.schema import Document, Generation, LLMResult

from mem0rylol.base.embeddings import BaseEmbeddings
from mem0rylol.base.llms import BaseLLM
from mem0rylol.base.vector_stores import BaseVectorStore
from mem0rylol.config import settings
from mem0rylol.embeddings.google_genai import GoogleGenAIEmbeddings
from mem0rylol.llms.cerebras import CerebrasLLM
from mem0rylol.memory.memory_manager import MemoryManager
from mem0rylol.memory.memory_types import Memory
from mem0rylol.schemas.base import BaseSchema
from mem0rylol.schemas.lancedb import LanceDBDocument, LanceDBSchema
from mem0rylol.vector_stores.lancedb import MemoryVectorStore


@pytest.fixture
def mock_embeddings():
    with patch("mem0rylol.embeddings.google_genai.GoogleGenerativeAIEmbeddings") as mock:
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm():
    with patch("mem0rylol.llms.cerebras.ChatCerebras") as mock:
        mock_instance = Mock()
        mock_instance.return_value = "Mocked response"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_vector_store():
    with patch("mem0rylol.vector_stores.lancedb.MemoryVectorStore") as mock_vector_store:
        mock_instance = MagicMock()
        mock_vector_store.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def memory_manager(mock_embeddings, mock_llm, mock_vector_store):
    embeddings = mock_embeddings
    llm = mock_llm
    vector_store = mock_vector_store
    return MemoryManager(
        table_name="test_memories",
        schema_cls=LanceDBSchema,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
    )


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

    result = memory_manager.max_marginal_relevance_search(
        "What are some popular programming languages?", k=3, fetch_k=5
    )
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(doc, Document) for doc in result)
    assert all("Programming language" in doc.page_content for doc in result)


def test_config_validation():
    assert settings.LANCEDB_CONNECTION_STRING is not None
    assert isinstance(settings.LANCEDB_CONNECTION_STRING, str)


def test_google_genai_embeddings():
    with patch("mem0rylol.embeddings.google_genai.GoogleGenerativeAIEmbeddings") as mock_embeddings:
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
    with patch("mem0rylol.llms.cerebras.ChatCerebras") as mock_cerebras:
        mock_instance = Mock()
        mock_instance.return_value = "Mocked response"
        mock_instance.generate.return_value = LLMResult(
            generations=[
                [Generation(text="Mocked generation 1")],
                [Generation(text="Mocked generation 2")],
            ]
        )
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
    with patch("mem0rylol.vector_stores.lancedb.lancedb") as mock_lancedb:
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

    # Test abstract methods
    with pytest.raises(NotImplementedError):
        BaseEmbeddings.embed_documents(None, ["test"])
    with pytest.raises(NotImplementedError):
        BaseEmbeddings.embed_query(None, "test")


def test_base_llm():
    class TestLLM(BaseLLM):
        def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            return "Test response"

        def generate(
            self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs
        ) -> LLMResult:
            return LLMResult(generations=[[Generation(text="Test generation")] for _ in prompts])

    llm = TestLLM()
    assert llm("Test prompt") == "Test response"
    result = llm.generate(["Prompt 1", "Prompt 2"])
    assert isinstance(result, LLMResult)
    assert len(result.generations) == 2
    assert result.generations[0][0].text == "Test generation"

    with pytest.raises(TypeError):
        BaseLLM()

    # Test abstract methods
    with pytest.raises(NotImplementedError):
        BaseLLM.__call__(None, "test")
    with pytest.raises(NotImplementedError):
        BaseLLM.generate(None, ["test"])


def test_base_vector_store():
    class TestVectorStore(BaseVectorStore):
        def create_table(self, table_name: str, schema: dict):
            return "Test table"

        def insert_data(self, table, data):
            pass

        def similarity_search(self, table, query: str, k: int = 4) -> List[Document]:
            return [Document(page_content="Test document")]

        def max_marginal_relevance_search(
            self, table, query: str, k: int = 4, fetch_k: int = 20
        ) -> List[Document]:
            return [Document(page_content="Test document")]

    vector_store = TestVectorStore()
    assert vector_store.create_table("test", {}) == "Test table"
    vector_store.insert_data("test_table", "test_data")
    assert len(vector_store.similarity_search("test_table", "query")) == 1
    assert len(vector_store.max_marginal_relevance_search("test_table", "query")) == 1

    with pytest.raises(TypeError):
        BaseVectorStore()

    # Test abstract methods
    with pytest.raises(NotImplementedError):
        BaseVectorStore.create_table(None, "test", {})
    with pytest.raises(NotImplementedError):
        BaseVectorStore.insert_data(None, "test", "data")
    with pytest.raises(NotImplementedError):
        BaseVectorStore.similarity_search(None, "test", "query")
    with pytest.raises(NotImplementedError):
        BaseVectorStore.max_marginal_relevance_search(None, "test", "query")


def test_memory_vector_store_connection():
    with patch("mem0rylol.vector_stores.lancedb.lancedb") as mock_lancedb:
        mock_connection = Mock()
        mock_lancedb.connect.return_value = mock_connection

        # Test HTTP connection
        with patch("mem0rylol.vector_stores.lancedb.settings") as mock_settings:
            mock_settings.LANCEDB_CONNECTION_STRING = "http://example.com"
            vector_store = MemoryVectorStore(Mock())
            assert vector_store.connection == mock_connection

        # Test local connection
        with (
            patch("mem0rylol.vector_stores.lancedb.settings") as mock_settings,
            patch("mem0rylol.vector_stores.lancedb.os") as mock_os,
        ):
            mock_settings.LANCEDB_CONNECTION_STRING = "./local_data"
            vector_store = MemoryVectorStore(Mock())
            mock_os.makedirs.assert_called_once_with("./local_data", exist_ok=True)
            assert vector_store.connection == mock_connection


def test_lancedb_schemas():
    # Test LanceDBSchema
    schema = LanceDBSchema(id="1", text="Test text", embedding=[0.1, 0.2, 0.3])
    assert schema.id == "1"
    assert schema.text == "Test text"
    assert schema.embedding == [0.1, 0.2, 0.3]

    # Test to_vector_store_schema method
    vector_store_schema = schema.to_vector_store_schema()
    assert isinstance(vector_store_schema, dict)
    assert vector_store_schema["id"] == "1"
    assert vector_store_schema["text"] == "Test text"
    assert vector_store_schema["embedding"] == [0.1, 0.2, 0.3]

    # Test LanceDBDocument
    doc = LanceDBDocument(
        id="2", text="Another test", embedding=[0.4, 0.5, 0.6], metadata={"key": "value"}
    )
    assert doc.id == "2"
    assert doc.text == "Another test"
    assert doc.embedding == [0.4, 0.5, 0.6]
    assert doc.metadata == {"key": "value"}

    # Test page_content property
    assert doc.page_content == "Another test"

    # Test LanceDBDocument without metadata
    doc_without_metadata = LanceDBDocument(id="3", text="No metadata", embedding=[0.7, 0.8, 0.9])
    assert doc_without_metadata.id == "3"
    assert doc_without_metadata.text == "No metadata"
    assert doc_without_metadata.embedding == [0.7, 0.8, 0.9]
    assert doc_without_metadata.metadata is None

    # Test LanceDBSchema model_config
    assert LanceDBSchema.model_config["arbitrary_types_allowed"] == True

    # Test LanceDBDocument.from_vector_store_schema method
    vector_store_doc = {
        "id": "4",
        "text": "Test from_vector_store_schema",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"key": "value"},
    }
    reconstructed_doc = LanceDBDocument.from_vector_store_schema(vector_store_doc)
    assert isinstance(reconstructed_doc, LanceDBDocument)
    assert reconstructed_doc.id == "4"
    assert reconstructed_doc.text == "Test from_vector_store_schema"
    assert reconstructed_doc.embedding == [0.1, 0.2, 0.3]
    assert reconstructed_doc.metadata == {"key": "value"}

    # Test LanceDBDocument.from_vector_store_schema method without metadata
    vector_store_doc_without_metadata = {
        "id": "5",
        "text": "No metadata",
        "embedding": [0.4, 0.5, 0.6],
    }
    reconstructed_doc_without_metadata = LanceDBDocument.from_vector_store_schema(
        vector_store_doc_without_metadata
    )
    assert isinstance(reconstructed_doc_without_metadata, LanceDBDocument)
    assert reconstructed_doc_without_metadata.id == "5"
    assert reconstructed_doc_without_metadata.text == "No metadata"
    assert reconstructed_doc_without_metadata.embedding == [0.4, 0.5, 0.6]
    assert reconstructed_doc_without_metadata.metadata is None
