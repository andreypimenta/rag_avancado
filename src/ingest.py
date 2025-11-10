from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.multimodal_processor import MultiModalProcessor
from pathlib import Path
from typing import List
import os

class DocumentProcessor:
    """Processa diferentes tipos de documentos com suporte multi-modal"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # Processador multi-modal
        self.multimodal = MultiModalProcessor()
        
        print("✅ Document Processor com Multi-modal inicializado")
    
    def load_document(self, file_path: str):
        """Carrega documento - tenta multi-modal primeiro"""
        extension = Path(file_path).suffix.lower()
        
        # Arquivos que o multi-modal processa melhor
        multimodal_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', 
                                '.xlsx', '.xls', '.csv', '.json'}
        
        if extension in multimodal_extensions:
            return self.multimodal.process_file(file_path)
        
        # PDF - usa multi-modal para OCR
        if extension == '.pdf':
            return self.multimodal.process_file(file_path)
        
        # Fallback para loaders tradicionais
        loaders = {
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
        }
        
        if extension in loaders:
            loader = loaders[extension](file_path)
            return loader.load()
        
        raise ValueError(f"Tipo de arquivo não suportado: {extension}")
    
    def process_documents(self, folder_path: str):
        """Processa todos os documentos de uma pasta"""
        documents = []
        
        for file_path in Path(folder_path).rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    docs = self.load_document(str(file_path))
                    
                    # Adiciona metadata adicional
                    for doc in docs:
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = file_path.name
                        if 'file_type' not in doc.metadata:
                            doc.metadata['file_type'] = file_path.suffix
                    
                    documents.extend(docs)
                except Exception as e:
                    print(f"⚠️  Erro ao processar {file_path.name}: {e}")
        
        # Faz chunking
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            print(f"\n✅ Total: {len(documents)} documentos → {len(chunks)} chunks")
            return chunks
        
        return []
