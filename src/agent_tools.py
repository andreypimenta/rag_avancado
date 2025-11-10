from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

class AgentTool(ABC):
    """Interface base para ferramentas do agent"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Executa a ferramenta"""
        pass

class DocumentSearchTool(AgentTool):
    """Tool para buscar nos documentos"""
    
    def __init__(self, rag_system):
        super().__init__(
            name="document_search",
            description="Busca informações nos documentos/conhecimento base. Use para perguntas sobre o conteúdo dos documentos carregados."
        )
        self.rag_system = rag_system
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Busca nos documentos"""
        print(f"   📄 [TOOL] Buscando documentos: '{query}'")
        
        try:
            answer = ""
            sources = []
            
            for chunk in self.rag_system.query_stream(query):
                if chunk['type'] == 'chunk':
                    answer += chunk['data']
                elif chunk['type'] == 'sources':
                    sources = chunk['data']
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "query": query,
                "tool": "document_search"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "document_search"
            }

class CalculatorTool(AgentTool):
    """Tool para cálculos matemáticos"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Realiza cálculos matemáticos. Use para operações como soma, subtração, multiplicação, divisão, percentuais, etc. Exemplo: '30 * 1500 / 100' para 30% de 1500."
        )
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """Calcula expressão"""
        print(f"   🔢 [TOOL] Calculando: '{expression}'")
        
        try:
            # Segurança básica
            allowed = set('0123456789+-*/()., ')
            if not all(c in allowed for c in expression):
                raise ValueError("Expressão contém caracteres inválidos")
            
            result = eval(expression)
            
            return {
                "success": True,
                "result": result,
                "expression": expression,
                "tool": "calculator"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "calculator"
            }

class DateTimeTool(AgentTool):
    """Tool para data/hora atual"""
    
    def __init__(self):
        super().__init__(
            name="datetime",
            description="Retorna data e hora atuais. Use quando precisar saber que dia/hora é hoje."
        )
    
    def execute(self) -> Dict[str, Any]:
        """Retorna data/hora"""
        print(f"   📅 [TOOL] Obtendo data/hora")
        
        try:
            now = datetime.now()
            
            return {
                "success": True,
                "datetime": now.isoformat(),
                "formatted": now.strftime("%d/%m/%Y %H:%M:%S"),
                "date": now.strftime("%d/%m/%Y"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "tool": "datetime"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "datetime"
            }

class KnowledgeTool(AgentTool):
    """Tool para conhecimento geral do LLM"""
    
    def __init__(self, llm):
        super().__init__(
            name="general_knowledge",
            description="Usa conhecimento geral do LLM. Use para perguntas gerais que NÃO estão nos documentos e NÃO são atuais (use web_search para info atual)."
        )
        self.llm = llm
    
    def execute(self, question: str) -> Dict[str, Any]:
        """Usa conhecimento do LLM"""
        print(f"   🧠 [TOOL] Conhecimento geral: '{question}'")
        
        try:
            prompt = f"Responda de forma direta e concisa:\n\n{question}"
            answer = self.llm.generate(prompt)
            
            return {
                "success": True,
                "answer": answer,
                "question": question,
                "tool": "general_knowledge"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "general_knowledge"
            }

class WebSearchTool(AgentTool):
    """Tool para buscar informações na web"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Busca informações na internet/web. Use quando a pergunta exige informações atuais, notícias, eventos recentes ou conhecimento que NÃO está nos documentos."
        )
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.has_brave = bool(self.brave_api_key)
        
        if self.has_brave:
            print("   ✅ Brave Search disponível")
        else:
            print("   ⚠️  Brave Search não configurado (usando fallback)")
    
    def execute(self, query: str, count: int = 3) -> Dict[str, Any]:
        """Busca na web"""
        print(f"   🌐 [TOOL] Buscando na web: '{query}'")
        
        if self.has_brave:
            return self._search_brave(query, count)
        else:
            return self._search_duckduckgo(query, count)
    
    def _search_brave(self, query: str, count: int) -> Dict[str, Any]:
        """Busca usando Brave Search API"""
        try:
            import requests
            
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }
            params = {
                "q": query,
                "count": count
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("web", {}).get("results", [])[:count]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("description", "")
                    })
                
                print(f"   ✅ Encontrados {len(results)} resultados")
                
                return {
                    "success": True,
                    "results": results,
                    "query": query,
                    "source": "brave_search",
                    "tool": "web_search"
                }
            else:
                return {
                    "success": False,
                    "error": f"Brave API error: {response.status_code}",
                    "tool": "web_search"
                }
                
        except Exception as e:
            print(f"   ❌ Erro Brave: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "web_search"
            }
    
    def _search_duckduckgo(self, query: str, count: int) -> Dict[str, Any]:
        """Busca usando DuckDuckGo (fallback)"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:count]:
                title_tag = result.find('a', class_='result__a')
                snippet_tag = result.find('a', class_='result__snippet')
                
                if title_tag:
                    results.append({
                        "title": title_tag.get_text(),
                        "url": title_tag.get('href', ''),
                        "description": snippet_tag.get_text() if snippet_tag else ""
                    })
            
            print(f"   ✅ Encontrados {len(results)} resultados")
            
            return {
                "success": True,
                "results": results,
                "query": query,
                "source": "duckduckgo",
                "tool": "web_search"
            }
            
        except Exception as e:
            print(f"   ❌ Erro DuckDuckGo: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "web_search"
            }

class ToolRegistry:
    """Registro de todas as tools disponíveis"""
    
    def __init__(self, rag_system, llm):
        self.tools = {
            "document_search": DocumentSearchTool(rag_system),
            "calculator": CalculatorTool(),
            "datetime": DateTimeTool(),
            "general_knowledge": KnowledgeTool(llm),
            "web_search": WebSearchTool()
        }
        
        print(f"✅ Tool Registry com {len(self.tools)} ferramentas")
    
    def get_tool(self, name: str) -> Optional[AgentTool]:
        """Retorna tool pelo nome"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Lista todas as tools"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]
