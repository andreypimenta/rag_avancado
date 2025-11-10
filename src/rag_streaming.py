from src.rag_advanced import AdvancedRAGSystem
from typing import Iterator, Optional

# Tenta importar cache
try:
    from src.cache_manager import CacheManager
    CACHE_AVAILABLE = True
except:
    CACHE_AVAILABLE = False

class StreamingRAGSystem(AdvancedRAGSystem):
    """RAG com streaming e cache opcional"""
    
    def __init__(self, vectorstore_manager, llm, k_documents=5, enable_cache=True):
        super().__init__(vectorstore_manager, llm, k_documents)
        
        self.enable_cache = enable_cache and CACHE_AVAILABLE
        if self.enable_cache:
            try:
                self.cache = CacheManager()
                print("✅ Cache ativado")
            except:
                self.enable_cache = False
                self.cache = None
                print("⚠️  Cache não disponível")
        else:
            self.cache = None
    
    def query_stream(
        self,
        question: str,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_expansion: bool = False
    ) -> Iterator[dict]:
        """Query com streaming e cache"""
        
        # Verifica cache
        if self.enable_cache and self.cache:
            cached = self.cache.get_response_cache(question)
            
            if cached:
                yield {'type': 'status', 'data': '💾 Cache HIT! Resposta instantânea'}
                
                for i in range(0, len(cached['answer']), 10):
                    yield {'type': 'chunk', 'data': cached['answer'][i:i+10]}
                
                yield {'type': 'metadata', 'data': {
                    'confidence': cached.get('confidence', 0),
                    'method': 'cached'
                }}
                yield {'type': 'sources', 'data': cached.get('sources', [])}
                return
        
        # Processa normalmente
        yield {'type': 'status', 'data': '🔍 Buscando documentos...'}
        
        queries = [question]
        if use_expansion:
            yield {'type': 'status', 'data': '🔄 Expandindo query...'}
            queries = self.advanced_retriever.query_expansion(question, self.llm)
        
        all_results = []
        for q in queries:
            if use_hybrid:
                results = self.advanced_retriever.hybrid_search(q, k=self.k_documents * 3)
            else:
                results = self.vectorstore.similarity_search_with_score(q, k=self.k_documents * 3)
            all_results.extend(results)
        
        seen = {}
        for doc, score in all_results:
            doc_id = doc.page_content[:100]
            if doc_id not in seen or score > seen[doc_id][1]:
                seen[doc_id] = (doc, score)
        
        unique_results = list(seen.values())
        unique_results.sort(key=lambda x: x[1], reverse=True)
        top_results = unique_results[:self.k_documents * 2]
        
        if use_rerank and len(top_results) > self.k_documents:
            yield {'type': 'status', 'data': '🎯 Reranking...'}
            docs = [doc for doc, _ in top_results]
            reranked = self.advanced_retriever.rerank_results(question, docs, top_k=self.k_documents)
            final_docs = [doc for doc, _ in reranked]
            final_scores = [score for _, score in reranked]
        else:
            final_docs = [doc for doc, _ in top_results[:self.k_documents]]
            final_scores = [score for _, score in top_results[:self.k_documents]]
        
        if not final_docs:
            yield {'type': 'chunk', 'data': "Não encontrei documentos relevantes."}
            return
        
        yield {'type': 'status', 'data': '🤖 Gerando resposta...\n'}
        
        contexts = [doc.page_content for doc in final_docs]
        prompt = self._create_prompt(question, contexts)
        
        full_answer = ""
        for chunk in self.llm.generate_stream(prompt):
            full_answer += chunk
            yield {'type': 'chunk', 'data': chunk}
        
        avg_confidence = sum(final_scores) / len(final_scores) if final_scores else 0.0
        
        metadata = {
            'confidence': float(avg_confidence),
            'method': f"streaming{'_hybrid' if use_hybrid else ''}{'_rerank' if use_rerank else ''}"
        }
        
        yield {'type': 'metadata', 'data': metadata}
        
        sources = [
            {"content": doc.page_content[:200] + "...", "metadata": doc.metadata, "score": float(score)}
            for doc, score in zip(final_docs, final_scores)
        ]
        
        yield {'type': 'sources', 'data': sources}
        
        # Salva no cache
        if self.enable_cache and self.cache:
            self.cache.set_response_cache(question, {
                'answer': full_answer,
                'confidence': float(avg_confidence),
                'method': metadata['method'],
                'sources': sources
            })
