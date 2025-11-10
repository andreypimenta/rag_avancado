from typing import List, Dict, Any, Iterator
from src.agent_tools import ToolRegistry
import json

class AutonomousAgent:
    """Agent autônomo com multi-step reasoning"""
    
    def __init__(self, rag_system, llm):
        self.rag_system = rag_system
        self.llm = llm
        self.tools = ToolRegistry(rag_system, llm)
        self.max_iterations = 3
        
        print("✅ Autonomous Agent inicializado")
    
    def _create_planning_prompt(self, question: str, tools_available: List[Dict], previous_results: List[Dict] = None) -> str:
        """Cria prompt para planejamento"""
        
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tools_available
        ])
        
        previous_context = ""
        if previous_results:
            previous_context = "\n\nResultados anteriores:\n" + "\n".join([
                f"- {r['tool']}: {json.dumps(r, ensure_ascii=False)[:200]}"
                for r in previous_results
            ])
        
        return f"""Você é um assistente IA que usa ferramentas em múltiplas etapas.

Ferramentas:
{tools_desc}

IMPORTANTE:
- Para percentuais no calculator: use "X * Y / 100"
- Se já calculou algo, pode usar document_search para buscar o resultado
- Use web_search para informações atuais, notícias, eventos recentes
- Analise se precisa de mais passos

Pergunta original: {question}
{previous_context}

Decida a próxima ação. Responda em JSON:
{{
    "reasoning": "análise do que fazer agora",
    "tool": "nome_da_ferramenta (ou null se terminou)",
    "tool_input": "entrada específica",
    "needs_tool": true/false,
    "is_complete": true/false
}}

Se já tem informação suficiente para responder, use is_complete: true.

JSON:"""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse resposta JSON"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            
            return {"needs_tool": False, "is_complete": True}
        except:
            return {"needs_tool": False, "is_complete": True}
    
    def _create_final_answer_prompt(self, question: str, tool_results: List[Dict]) -> str:
        """Cria prompt para resposta final"""
        
        results_text = ""
        for r in tool_results:
            tool_name = r.get('tool', 'unknown')
            
            if tool_name == 'calculator' and r.get('success'):
                results_text += f"\n✓ Cálculo: {r.get('expression')} = {r.get('result')}"
            
            elif tool_name == 'document_search' and r.get('success'):
                results_text += f"\n✓ Documentos: {r.get('answer', '')[:300]}"
            
            elif tool_name == 'datetime' and r.get('success'):
                results_text += f"\n✓ Data/Hora: {r.get('formatted', '')}"
            
            elif tool_name == 'general_knowledge' and r.get('success'):
                results_text += f"\n✓ Conhecimento: {r.get('answer', '')[:200]}"
            
            elif tool_name == 'web_search' and r.get('success'):
                web_results = r.get('results', [])
                results_text += f"\n✓ Web Search ({len(web_results)} resultados):"
                for idx, item in enumerate(web_results[:3], 1):
                    results_text += f"\n  {idx}. {item.get('title', '')}"
                    results_text += f"\n     {item.get('description', '')[:150]}"
                    results_text += f"\n     URL: {item.get('url', '')}"
        
        return f"""Com base nos resultados, responda de forma completa e natural.

Pergunta: {question}

Resultados das ferramentas:{results_text}

Sua resposta final (clara, direta e útil):"""
    
    def process_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """Processa com múltiplas iterações"""
        
        yield {'type': 'status', 'data': '🧠 Agent iniciando análise...'}
        
        tool_results = []
        tools_list = self.tools.list_tools()
        
        # Multi-step loop
        for iteration in range(self.max_iterations):
            
            # Planning
            planning_prompt = self._create_planning_prompt(
                question, 
                tools_list, 
                tool_results if tool_results else None
            )
            
            try:
                plan_response = self.llm.generate(planning_prompt)
                plan = self._parse_json_response(plan_response)
                
                # Mostra reasoning
                if 'reasoning' in plan and iteration == 0:
                    yield {
                        'type': 'reasoning',
                        'data': f"💭 {plan['reasoning']}"
                    }
                
                # Verifica se terminou
                if plan.get('is_complete', False) or not plan.get('needs_tool', False):
                    break
                
                # Executa tool
                if 'tool' in plan and plan['tool']:
                    tool_name = plan['tool']
                    tool_input = plan.get('tool_input', question)
                    
                    yield {
                        'type': 'status',
                        'data': f"🔧 Passo {iteration + 1}: {tool_name}"
                    }
                    
                    tool = self.tools.get_tool(tool_name)
                    
                    if tool:
                        # Executa a tool apropriada
                        if tool_name == "document_search":
                            result = tool.execute(tool_input)
                        elif tool_name == "calculator":
                            result = tool.execute(tool_input)
                        elif tool_name == "datetime":
                            result = tool.execute()
                        elif tool_name == "general_knowledge":
                            result = tool.execute(tool_input)
                        elif tool_name == "web_search":
                            result = tool.execute(tool_input)
                        else:
                            result = {"success": False, "error": "Tool desconhecida"}
                        
                        tool_results.append(result)
                        
                        if result.get('success'):
                            yield {'type': 'tool_result', 'data': f"✅ OK"}
                        else:
                            yield {'type': 'tool_result', 'data': f"⚠️ Erro"}
                            break
                else:
                    break
                    
            except Exception as e:
                print(f"❌ Erro na iteração {iteration}: {e}")
                break
        
        # Resposta final
        yield {'type': 'status', 'data': '📝 Sintetizando resposta...\n'}
        
        if tool_results:
            final_prompt = self._create_final_answer_prompt(question, tool_results)
        else:
            final_prompt = f"Responda:\n\n{question}"
        
        # Stream
        full_answer = ""
        for chunk in self.llm.generate_stream(final_prompt):
            full_answer += chunk
            yield {'type': 'chunk', 'data': chunk}
        
        # Metadata
        yield {
            'type': 'metadata',
            'data': {
                'agent_mode': True,
                'tools_used': [r['tool'] for r in tool_results],
                'iterations': len(tool_results)
            }
        }
        
        # Sources
        sources = []
        for result in tool_results:
            if result.get('tool') == 'document_search' and 'sources' in result:
                sources.extend(result['sources'])
        
        if sources:
            yield {'type': 'sources', 'data': sources}
