from src.rag import RAGSystem
from src.advanced_retriever import AdvancedRetriever
from typing import Optional

class AdvancedRAGSystem(RAGSystem):
    """RAG com retrieval avançado"""
    
    def __init__(self, vectorstore_manager, llm, k_documents=5):
        super().__init__(vectorstore_manager, llm, k_documents)
        self.advanced_retriever = AdvancedRetriever(vectorstore_manager)
        print("✅ Retrieval avançado ativado!")
    
    def query_advanced(
        self,
        question: str,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_expansion: bool = False,
        return_sources: bool = True
    ) -> dict:
        """Query com features avançadas"""
        queries = [question]
        
        if use_expansion:
            print("🔄 Expandindo query...")
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
            print("🎯 Aplicando reranking...")
            docs = [doc for doc, _ in top_results]
            reranked = self.advanced_retriever.rerank_results(question, docs, top_k=self.k_documents)
            final_docs = [doc for doc, _ in reranked]
            final_scores = [score for _, score in reranked]
        else:
            final_docs = [doc for doc, _ in top_results[:self.k_documents]]
            final_scores = [score for _, score in top_results[:self.k_documents]]
        
        if not final_docs:
            return {
                "answer": "Não encontrei documentos relevantes.",
                "sources": [],
                "confidence": 0.0,
                "method": "advanced"
            }
        
        print("🤖 Gerando resposta...")
        contexts = [doc.page_content for doc in final_docs]
        prompt = self._create_prompt(question, contexts)
        answer = self.llm.generate(prompt)
        
        avg_confidence = sum(final_scores) / len(final_scores) if final_scores else 0.0
        
        result = {
            "answer": answer,
            "confidence": float(avg_confidence),
            "method": f"advanced{'_hybrid' if use_hybrid else ''}{'_rerank' if use_rerank else ''}{'_expansion' if use_expansion else ''}"
        }
        
        if return_sources:
            result["sources"] = [
                {"content": doc.page_content[:200] + "...", "metadata": doc.metadata, "score": float(score)}
                for doc, score in zip(final_docs, final_scores)
            ]
        
        return result
    
    def query(self, question: str, **kwargs):
        """Query usando retrieval avançado"""
        return self.query_advanced(question, use_hybrid=True, use_rerank=True, use_expansion=False, **kwargs)
