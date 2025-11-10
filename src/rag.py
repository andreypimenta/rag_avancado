from typing import List, Optional
from src.vectorstore import VectorStoreManager
from src.llm import BaseLLM

class RAGSystem:
    """Sistema RAG completo"""
    
    def __init__(self, vectorstore_manager: VectorStoreManager, llm: BaseLLM, k_documents: int = 5):
        self.vectorstore = vectorstore_manager
        self.llm = llm
        self.k_documents = k_documents
    
    def _create_prompt(self, query: str, contexts: List[str]) -> str:
        """Cria o prompt para o LLM"""
        context_text = "\n\n---\n\n".join(
            [f"[Documento {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
        )
        
        prompt = f"""Baseado APENAS nos documentos fornecidos abaixo, responda a pergunta do usuário.

DOCUMENTOS:
{context_text}

PERGUNTA: {query}

INSTRUÇÕES:
- Use APENAS informações dos documentos fornecidos
- Cite as fontes usando [Documento X]
- Se a informação não estiver nos documentos, diga "Não encontrei essa informação nos documentos fornecidos"
- Seja preciso e objetivo

RESPOSTA:"""
        
        return prompt
    
    def query(self, question: str, filter_metadata: Optional[dict] = None, return_sources: bool = True) -> dict:
        """Consulta o sistema RAG"""
        
        print(f"🔍 Buscando documentos relevantes...")
        results = self.vectorstore.similarity_search_with_score(question, k=self.k_documents)
        
        # Removido o filtro de score mínimo - aceita todos os resultados
        filtered_results = results
        
        if not filtered_results:
            return {
                "answer": "Não encontrei documentos relevantes para sua pergunta.",
                "sources": [],
                "confidence": 0.0
            }
        
        contexts = [doc.page_content for doc, _ in filtered_results]
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in filtered_results
        ]
        
        print(f"🤖 Gerando resposta...")
        prompt = self._create_prompt(question, contexts)
        answer = self.llm.generate(prompt)
        
        avg_confidence = sum(score for _, score in filtered_results) / len(filtered_results)
        
        result = {
            "answer": answer,
            "confidence": float(avg_confidence),
        }
        
        if return_sources:
            result["sources"] = sources
        
        return result
    
    def conversational_query(self, question: str, chat_history: List[dict] = None):
        """Query com contexto de conversa"""
        if chat_history:
            context = "\n".join([
                f"User: {msg['question']}\nAssistant: {msg['answer']}"
                for msg in chat_history[-3:]
            ])
            
            reformulated = f"""Dado o histórico de conversa:
{context}

Nova pergunta: {question}

Reformule a pergunta para ser autocontida:"""
            
            question = self.llm.generate(reformulated)
        
        return self.query(question)
