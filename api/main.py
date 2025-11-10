from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from pathlib import Path
import asyncio
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).parent.parent))

from src.ingest import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.llm import LLMFactory
from src.rag_streaming import StreamingRAGSystem
from src.autonomous_agent import AutonomousAgent
from src.auth import get_current_user
from src.database import get_db
from src.database.models import User
from dotenv import load_dotenv
import shutil

load_dotenv()

app = FastAPI(title="RAG Platform API", version="3.0.0 - Multi-Tenant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global (para compatibilidade com código legado)
rag_system = None
agent_system = None
vs_manager = None
whatsapp_bot = None

class QueryRequest(BaseModel):
    question: str
    use_hybrid: bool = True
    use_rerank: bool = True
    use_expansion: bool = False
    use_agent: bool = False

class QueryResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    method: str
    sources: List[dict] = []
    agent_mode: bool = False
    tools_used: List[str] = []

@app.on_event("startup")
async def startup_event():
    """Inicializa o sistema RAG + Agent"""
    global rag_system, agent_system, vs_manager
    
    print("🚀 Inicializando RAG Platform API v3.0...")
    
    vs_manager = VectorStoreManager(persist_directory="./vectorstore")
    
    if Path("./vectorstore").exists():
        vs_manager.load_vectorstore()
        print("✅ Vector store carregado")
    else:
        print("⚠️  Vector store não encontrado")
    
    llm = LLMFactory.create("groq")
    
    rag_system = StreamingRAGSystem(
        vectorstore_manager=vs_manager,
        llm=llm,
        k_documents=5
    )
    
    agent_system = AutonomousAgent(rag_system, llm)
    
    print("✅ RAG + Agent API pronta!")

@app.on_event("startup")
async def startup_whatsapp():
    """Inicializa WhatsApp Bot"""
    global whatsapp_bot
    
    try:
        from src.whatsapp_bot import WhatsAppBot
        
        if agent_system and rag_system:
            whatsapp_bot = WhatsAppBot(
                rag_system=rag_system,
                agent_system=agent_system
            )
            print("✅ WhatsApp Bot pronto")
            print("📱 Conectar: http://localhost:8000/whatsapp/connect")
    except Exception as e:
        print(f"⚠️  WhatsApp Bot erro: {e}")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "RAG Platform API - Multi-Tenant",
        "version": "3.0.0",
        "features": ["rag", "agent", "cache", "multimodal", "whatsapp", "multi-tenant", "auth"]
    }

# ==================== RAG ENDPOINTS (LEGADO) ====================

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        answer_text = ""
        metadata = {}
        sources = []
        
        if request.use_agent and agent_system:
            stream = agent_system.process_stream(request.question)
        else:
            stream = rag_system.query_stream(
                request.question,
                request.use_hybrid,
                request.use_rerank,
                request.use_expansion
            )
        
        for chunk in stream:
            if chunk['type'] == 'chunk':
                answer_text += chunk['data']
            elif chunk['type'] == 'metadata':
                metadata = chunk['data']
            elif chunk['type'] == 'sources':
                sources = chunk['data']
        
        return QueryResponse(
            answer=answer_text,
            confidence=metadata.get('confidence', 0.0),
            method=metadata.get('method', 'agent' if request.use_agent else 'rag'),
            sources=sources,
            agent_mode=metadata.get('agent_mode', False),
            tools_used=metadata.get('tools_used', [])
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"\n📤 Upload: {file.filename}")
        
        content = await file.read()
        file_size = len(content)
        print(f"   📊 Tamanho: {file_size / 1024 / 1024:.2f}MB")
        
        upload_dir = Path("./data")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        
        with file_path.open("wb") as buffer:
            buffer.write(content)
        
        print(f"   💾 Salvo: {file_path}")
        print(f"   🔄 Processando...")
        
        processor = DocumentProcessor()
        docs = processor.load_document(str(file_path))
        
        if not docs:
            raise HTTPException(status_code=400, detail="Nenhum conteúdo extraído")
        
        print(f"   ✅ Documentos: {len(docs)}")
        
        chunks = processor.text_splitter.split_documents(docs)
        print(f"   ✅ Chunks: {len(chunks)}")
        
        if vs_manager and vs_manager.vectorstore:
            vs_manager.vectorstore.add_documents(chunks)
            print(f"   ✅ Adicionado ao vector store\n")
        
        return {
            "status": "success",
            "filename": file.filename,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "documents_extracted": len(docs),
            "chunks_added": len(chunks),
            "message": f"✅ {len(chunks)} chunks adicionados"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    if not vs_manager or not vs_manager.vectorstore:
        return {"status": "not_initialized"}
    
    try:
        collection = vs_manager.vectorstore.get()
        
        return {
            "status": "ok",
            "total_documents": len(collection.get('ids', [])),
            "vectorstore_path": vs_manager.persist_directory,
            "agent_available": agent_system is not None,
            "whatsapp_available": whatsapp_bot is not None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/cache/stats")
async def get_cache_stats():
    if not rag_system or not hasattr(rag_system, 'cache') or not rag_system.cache:
        return {"status": "cache_disabled"}
    
    try:
        stats = rag_system.cache.get_stats()
        sizes = rag_system.cache.get_cache_size()
        
        return {
            "status": "ok",
            **stats,
            **sizes
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/cache/clear")
async def clear_cache():
    if not rag_system or not hasattr(rag_system, 'cache') or not rag_system.cache:
        raise HTTPException(status_code=404, detail="Cache não disponível")
    
    try:
        rag_system.cache.clear_cache()
        return {"status": "success", "message": "Cache limpo"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WHATSAPP BOT ====================

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(data: dict):
    """Recebe webhooks do Evolution API"""
    if not whatsapp_bot:
        raise HTTPException(status_code=503, detail="WhatsApp Bot offline")
    
    try:
        await whatsapp_bot.handle_webhook(data)
        return {"status": "ok"}
    except Exception as e:
        print(f"❌ Erro webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/whatsapp/connect")
async def whatsapp_connect():
    """Conecta WhatsApp (gera QR Code)"""
    if not whatsapp_bot:
        raise HTTPException(status_code=503, detail="WhatsApp Bot offline")
    
    try:
        print("\n📱 Criando instância WhatsApp...")
        await whatsapp_bot.create_instance()
        
        await asyncio.sleep(3)
        
        print("📱 Obtendo QR Code...")
        await whatsapp_bot.get_qrcode()
        
        return {
            "status": "ok",
            "message": "Escaneie whatsapp_qrcode.png com seu WhatsApp",
            "file": "whatsapp_qrcode.png",
            "instructions": [
                "1. Abra WhatsApp no celular",
                "2. Configurações > Aparelhos conectados",
                "3. Conectar um aparelho",
                "4. Escaneie whatsapp_qrcode.png"
            ]
        }
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/whatsapp/status")
async def whatsapp_status():
    """Status da conexão WhatsApp"""
    if not whatsapp_bot:
        return {"status": "offline", "message": "Bot não inicializado"}
    
    return {
        "status": "online",
        "instance": whatsapp_bot.instance_name,
        "evolution_url": whatsapp_bot.evolution_url
    }

# ==================== MULTI-TENANT ROUTES ====================

from api.auth_routes import router as auth_router
from api.assistant_routes import router as assistant_router

app.include_router(auth_router)
app.include_router(assistant_router)

print("✅ Rotas multi-tenant carregadas")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
