from typing import List, Tuple, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

class AdvancedRetriever:
    """Retrieval avançado com múltiplas estratégias"""
    
    def __init__(self, vectorstore_manager):
        self.vectorstore = vectorstore_manager
        self.bm25 = None
        self.documents = []
        self.reranker_available = False
        self._initialize_bm25()
        self._check_reranker()
    
    def _check_reranker(self):
        """Verifica se sentence-transformers está disponível"""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker_available = True
            print("✅ Reranking disponível")
        except ImportError:
            self.reranker_available = False
            print("⚠️  Reranking não disponível")
    
    def _initialize_bm25(self):
        """Inicializa BM25 para keyword search"""
        try:
            all_docs = self.vectorstore.vectorstore.get()
            
            if all_docs and 'documents' in all_docs:
                self.documents = []
                for i, doc_text in enumerate(all_docs['documents']):
                    metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                    doc = Document(page_content=doc_text, metadata=metadata)
                    self.documents.append(doc)
                
                tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                print(f"✅ BM25 inicializado com {len(self.documents)} documentos")
        except Exception as e:
            print(f"⚠️  Erro ao inicializar BM25: {e}")
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10,
        vector_weight: float = 0.7,
        filter_metadata: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Hybrid search: combina busca vetorial + keyword (BM25)"""
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)
        keyword_results = self._bm25_search(query, k=k * 2)
        combined = self._combine_results(vector_results, keyword_results, vector_weight, filter_metadata)
        return combined[:k]
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Busca usando BM25"""
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def _combine_results(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        vector_weight: float,
        filter_metadata: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Combina resultados com normalização"""
        def normalize(results):
            if not results:
                return []
            scores = [s for _, s in results]
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            score_range = max_score - min_score if max_score != min_score else 1
            return [(doc, (s - min_score) / score_range) for doc, s in results]
        
        vector_norm = normalize(vector_results)
        keyword_norm = normalize(keyword_results)
        
        combined_dict = {}
        keyword_weight = 1 - vector_weight
        
        for doc, score in vector_norm:
            doc_id = doc.page_content[:100]
            combined_dict[doc_id] = {'doc': doc, 'score': score * vector_weight}
        
        for doc, score in keyword_norm:
            doc_id = doc.page_content[:100]
            if doc_id in combined_dict:
                combined_dict[doc_id]['score'] += score * keyword_weight
            else:
                combined_dict[doc_id] = {'doc': doc, 'score': score * keyword_weight}
        
        if filter_metadata:
            filtered_dict = {}
            for doc_id, data in combined_dict.items():
                doc = data['doc']
                if all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    filtered_dict[doc_id] = data
            combined_dict = filtered_dict
        
        result = [(data['doc'], data['score']) for data in combined_dict.values()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def rerank_results(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Reranking com cross-encoder"""
        if not self.reranker_available:
            return [(doc, 1.0 - (i * 0.1)) for i, doc in enumerate(documents[:top_k])]
        
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            return doc_score_pairs[:top_k]
        except Exception as e:
            print(f"⚠️  Erro no reranking: {e}")
            return [(doc, 1.0 - (i * 0.1)) for i, doc in enumerate(documents[:top_k])]
    
    def query_expansion(self, query: str, llm) -> List[str]:
        """Expande query em variações"""
        prompt = f"""Gere 2 variações diferentes da seguinte pergunta:

Pergunta: {query}

Variação 1:"""
        
        try:
            response = llm.generate(prompt)
            variations = [query]
            lines = response.strip().split('\n')
            
            for line in lines:
                clean = line.strip()
                for prefix in ['Variação 1:', 'Variação 2:', '1.', '2.', '-']:
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):].strip()
                
                if clean and len(clean) > 10 and clean not in variations:
                    variations.append(clean)
            
            return variations[:3]
        except Exception as e:
            print(f"⚠️  Erro na expansão: {e}")
            return [query]
