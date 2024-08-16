import os
os.environ['PINECONE_API_KEY'] = "73bdec39-1f93-47fc-bd2f-f02883d7be83"
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_api_key

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

class VectorDB:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings()
        self.db = self.__build_db__()

    def __build_db__(self):
        db = PineconeVectorStore.from_existing_index(
            index_name="docs-rag-chatbot",
            namespace="docs-ute_store",
            embedding=self.embedding
        )
        return db
    
    def get_retriever(self,
                      search_type: str = 'similarity', 
                      search_kwargs: dict = {'k': 10}):
        retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        return retriever


