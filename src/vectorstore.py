from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Tuple
import os

class VectorStoreManager:
    """Gerencia o banco vetorial"""
    
    def __init__(self, persist_directory="./vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
    
    def create_vectorstore(self, chunks):
        """Cria o vector store a partir dos chunks"""
        print("🔄 Criando vector store...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✅ Vector store criado com {len(chunks)} chunks")
        return self.vectorstore
    
    def load_vectorstore(self):
        """Carrega vector store existente"""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError("Vector store não encontrado.")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        print("✅ Vector store carregado")
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[dict] = None):
        """Busca por similaridade"""
        if not self.vectorstore:
            raise ValueError("Vector store não inicializado")
        
        return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Busca com scores de relevância (corrigido para ChromaDB)"""
        if not self.vectorstore:
            raise ValueError("Vector store não inicializado")
        
        # ChromaDB retorna distâncias (menores = mais similares)
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Converte distâncias para scores de similaridade (0-1)
        converted_results = []
        for doc, distance in results:
            # Converte distância euclidiana para similaridade
            similarity = 1 / (1 + abs(distance))
            converted_results.append((doc, similarity))
        
        return converted_results
