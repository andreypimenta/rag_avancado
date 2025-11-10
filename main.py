from src.ingest import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.llm import LLMFactory
from src.rag_streaming import StreamingRAGSystem
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys
from pathlib import Path
import time

load_dotenv()
console = Console()

class RAGInterface:
    def __init__(self):
        self.rag_system = None
        self.chat_history = []
    
    def setup(self):
        console.print("\n" + "="*60, style="bold cyan")
        console.print("🚀 RAG UNIVERSAL - Setup Avançado + Streaming", style="bold cyan")
        console.print("="*60 + "\n", style="bold cyan")
        
        vectorstore_path = "./vectorstore"
        needs_creation = not Path(vectorstore_path).exists()
        
        if needs_creation:
            console.print("📁 Vector store não encontrado. Vamos criar um!", style="yellow")
            data_folder = input("Caminho da pasta com documentos [./data]: ").strip() or "./data"
            
            if not Path(data_folder).exists():
                console.print(f"❌ Pasta {data_folder} não existe!", style="red")
                sys.exit(1)
            
            console.print("\n🔄 Processando documentos...", style="yellow")
            processor = DocumentProcessor()
            chunks = processor.process_documents(data_folder)
            
            if not chunks:
                console.print("❌ Nenhum documento encontrado!", style="red")
                sys.exit(1)
            
            vs_manager = VectorStoreManager(persist_directory=vectorstore_path)
            vs_manager.create_vectorstore(chunks)
        else:
            console.print("✅ Vector store existente encontrado!", style="green")
            vs_manager = VectorStoreManager(persist_directory=vectorstore_path)
            vs_manager.load_vectorstore()
        
        console.print("\n🤖 Escolha o modelo de linguagem:", style="bold")
        console.print("1. Groq (Llama 3.3 70B) - [yellow]SUPER RÁPIDO![/yellow] ⚡⚡⚡")
        console.print("2. Google Gemini - [green]Rápido[/green]")
        console.print("3. DeepSeek - [cyan]Normal[/cyan]")
        console.print("4. Claude (Anthropic) - [blue]Normal (melhor streaming visual)[/blue]")
        console.print("5. GPT-4 (OpenAI) - [magenta]Mais lento (melhor pra ver streaming)[/magenta]")
        console.print("6. GPT-3.5 (OpenAI) - [magenta]Rápido[/magenta]")
        
        llm_choice = input("\nEscolha (1-6): ").strip()
        
        llm_configs = {
            "1": ("groq", {"model": "llama-3.3-70b-versatile"}),
            "2": ("gemini", {"model": "gemini-pro"}),
            "3": ("deepseek", {}),
            "4": ("claude", {}),
            "5": ("openai", {"model": "gpt-4"}),
            "6": ("openai", {"model": "gpt-3.5-turbo"}),
        }
        
        provider, config = llm_configs.get(llm_choice, llm_configs["1"])
        
        try:
            llm = LLMFactory.create(provider, **config)
            console.print(f"\n✅ Usando {provider.upper()}", style="green")
        except Exception as e:
            console.print(f"\n❌ Erro ao configurar {provider}: {e}", style="red")
            console.print("💡 Verifique se a API key está no .env", style="yellow")
            sys.exit(1)
        
        self.rag_system = StreamingRAGSystem(
            vectorstore_manager=vs_manager,
            llm=llm,
            k_documents=5
        )
        
        console.print("\n✅ Sistema RAG com Streaming configurado!\n", style="green bold")
    
    def interactive_mode(self):
        console.print(Panel.fit(
            "[bold cyan]💬 MODO INTERATIVO COM STREAMING[/bold cyan]\n\n"
            "[yellow]Comandos:[/yellow]\n"
            "  • Digite sua pergunta\n"
            "  • 'avançado' - Query expansion\n"
            "  • 'lento' - Adiciona delay visual (0.01s/chunk)\n"
            "  • 'rápido' - Remove delay\n"
            "  • 'sair' - Encerrar",
            border_style="cyan"
        ))
        
        use_expansion = False
        visual_delay = 0.0  # Delay artificial
        
        while True:
            try:
                question = console.input("\n[bold blue]👤 Você:[/bold blue] ").strip()
                
                if not question:
                    continue
                
                if question.lower() == "sair":
                    console.print("\n👋 Até logo!", style="bold green")
                    break
                
                if question.lower() == "avançado":
                    use_expansion = not use_expansion
                    status = "ATIVADA" if use_expansion else "DESATIVADA"
                    console.print(f"🔄 Query expansion {status}", style="yellow")
                    continue
                
                if question.lower() == "lento":
                    visual_delay = 0.01
                    console.print("🐢 Modo lento ativado (delay visual)", style="yellow")
                    continue
                
                if question.lower() == "rápido":
                    visual_delay = 0.0
                    console.print("⚡ Modo rápido ativado", style="yellow")
                    continue
                
                # Streaming da resposta
                console.print()
                answer_text = ""
                metadata = {}
                sources = []
                start_time = time.time()
                token_count = 0
                
                console.print("[bold green]🤖 Assistente:[/bold green]\n", end='')
                
                for chunk in self.rag_system.query_stream(
                    question,
                    use_hybrid=True,
                    use_rerank=True,
                    use_expansion=use_expansion
                ):
                    if chunk['type'] == 'status':
                        console.print(chunk['data'], style="dim italic")
                    
                    elif chunk['type'] == 'chunk':
                        console.print(chunk['data'], end='', style="bold white")
                        answer_text += chunk['data']
                        token_count += 1
                        
                        # Delay visual opcional
                        if visual_delay > 0:
                            time.sleep(visual_delay)
                        
                        sys.stdout.flush()
                    
                    elif chunk['type'] == 'metadata':
                        metadata = chunk['data']
                    
                    elif chunk['type'] == 'sources':
                        sources = chunk['data']
                
                elapsed_time = time.time() - start_time
                tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
                
                # Exibe metadata com estatísticas
                console.print("\n")
                console.print(Panel(
                    f"[cyan]📊 Confiança:[/cyan] {metadata.get('confidence', 0):.2%}\n"
                    f"[cyan]⚙️  Método:[/cyan] {metadata.get('method', 'unknown')}\n"
                    f"[cyan]⚡ Velocidade:[/cyan] ~{tokens_per_sec:.0f} tokens/seg\n"
                    f"[cyan]⏱️  Tempo:[/cyan] {elapsed_time:.2f}s",
                    border_style="cyan",
                    title="[bold]Estatísticas[/bold]"
                ))
                
                # Exibe fontes
                if sources:
                    console.print(f"\n[bold yellow]📚 Fontes ({len(sources)}):[/bold yellow]")
                    for i, source in enumerate(sources[:3], 1):
                        score = source['score']
                        score_emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                        console.print(f"  {score_emoji} [{i}] {source['metadata'].get('source', 'Unknown')} [dim](score: {score:.2f})[/dim]")
                
                self.chat_history.append({
                    "question": question,
                    "answer": answer_text
                })
                
            except KeyboardInterrupt:
                console.print("\n\n👋 Até logo!", style="bold green")
                break
            except Exception as e:
                console.print(f"\n❌ Erro: {e}", style="red")
                import traceback
                traceback.print_exc()

def main():
    interface = RAGInterface()
    interface.setup()
    interface.interactive_mode()

if __name__ == "__main__":
    main()
