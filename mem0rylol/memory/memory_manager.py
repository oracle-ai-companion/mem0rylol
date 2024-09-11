import uuid
from typing import List, Optional, Type, Dict, Tuple, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate

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

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        @brief Perform a similarity search on the memory vector store.
        @param query The query to search for.
        @param k The number of results to return.
        @return A list of tuples containing the matching Documents and their similarity scores.
        """
        results = self.vector_store.similarity_search_with_score(self.table, query, k)
        return [(Document(page_content=doc.page_content, metadata={"id": doc.metadata["id"]}), score) for doc, score in results]

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

    MEMORY_DEDUCTION_PROMPT = PromptTemplate(
        input_variables=["user_input", "metadata"],
        template="""Deduce the facts, preferences, and memories from the provided text.
Just return the facts, preferences, and memories in bullet points:
Natural language text: {user_input}
User/Agent details: {metadata}
Constraint for deducing facts, preferences, and memories:
- The facts, preferences, and memories should be concise and informative.
- Don't start by "The person likes Pizza". Instead, start with "Likes Pizza".
- Don't remember the user/agent details provided. Only remember the facts, preferences, and memories.
Deduced facts, preferences, and memories:"""
    )

    def extract_memories(self, user_input: str, metadata: Dict[str, Any]) -> List[Memory]:
        prompt = self.MEMORY_DEDUCTION_PROMPT.format(user_input=user_input, metadata=metadata)
        response = self.llm(prompt)
        
        # Split the response into individual memories
        memory_texts = [m.strip() for m in response.split('\n') if m.strip()]
        
        # Create Memory objects
        memories = [Memory(text=text) for text in memory_texts]
        
        return memories

    UPDATE_MEMORY_PROMPT = PromptTemplate(
        input_variables=["existing_memories", "memory"],
        template="""You are an expert at merging, updating, and organizing memories. When provided with existing memories and new information, your task is to merge and update the memory list to reflect the most accurate and current information. You are also provided with the matching score for each existing memory to the new information. Make sure to leverage this information to make informed decisions about which memories to update or merge.
Guidelines:
- Eliminate duplicate memories and merge related memories to ensure a concise and updated list.
- If a memory is directly contradicted by new information, critically evaluate both pieces of information:
    - If the new memory provides a more recent or accurate update, replace the old memory with new one.
    - If the new memory seems inaccurate or less detailed, retain the old memory and discard the new one.
- Maintain a consistent and clear style throughout all memories, ensuring each entry is concise yet informative.
- If the new memory is a variation or extension of an existing memory, update the existing memory to reflect the new information.
Here are the details of the task:
- Existing Memories:
{existing_memories}
- New Memory: {memory}"""
    )

    def update_memories(self, new_memory: Memory, similar_memories: List[Tuple[Document, float]]) -> List[Memory]:
        existing_memories = "\n".join([f"- {doc.page_content} (Similarity: {score:.2f})" for doc, score in similar_memories])
        prompt = self.UPDATE_MEMORY_PROMPT.format(existing_memories=existing_memories, memory=new_memory.text)
        
        response = self.llm(prompt)
        
        # Split the response into individual updated memories
        updated_memory_texts = [m.strip() for m in response.split('\n') if m.strip()]
        
        # Create new Memory objects
        updated_memories = [Memory(text=text) for text in updated_memory_texts]
        
        # Update the vector store
        for memory in updated_memories:
            self.add_memory(memory)
        
        # Remove old memories that were updated
        for doc, _ in similar_memories:
            self.vector_store.delete(self.table, [doc.metadata['id']])
        
        return updated_memories

    MEMORY_ANSWER_PROMPT = PromptTemplate(
        input_variables=["question", "memories"],
        template="""You are an expert at answering questions based on the provided memories. Your task is to provide accurate and concise answers to the questions by leveraging the information given in the memories.
Guidelines:
- Extract relevant information from the memories based on the question.
- If no relevant information is found, make sure you don't say no information is found. Instead, accept the question and provide a general response.
- Ensure that the answers are clear, concise, and directly address the question.
Here are the details of the task:
Question: {question}
Relevant Memories:
{memories}
Answer:"""
    )

    def generate_response(self, question: str) -> str:
        relevant_memories = self.similarity_search(question)
        memories_text = "\n".join([f"- {doc.page_content}" for doc, _ in relevant_memories])
        
        prompt = self.MEMORY_ANSWER_PROMPT.format(question=question, memories=memories_text)
        response = self.llm(prompt)
        
        return response
