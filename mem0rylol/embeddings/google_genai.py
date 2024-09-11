from langchain_google_genai import GoogleGenerativeAIEmbeddings

from mem0rylol.base.embeddings import BaseEmbeddings


class GoogleGenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model="models/text-embedding-004"):
        self.model = GoogleGenerativeAIEmbeddings(model=model)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)
