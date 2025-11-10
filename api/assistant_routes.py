from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
import shutil
from datetime import datetime

from src.database import get_db
from src.database.models import User, Assistant, Document
from src.auth import get_current_user
from src.schemas import AssistantCreate, AssistantResponse, AssistantUpdate, DocumentResponse

router = APIRouter(prefix="/assistants", tags=["Assistants"])

@router.post("", response_model=AssistantResponse, status_code=status.HTTP_201_CREATED)
async def create_assistant(
    assistant_data: AssistantCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cria novo assistente"""
    
    new_assistant = Assistant(
        user_id=current_user.id,
        **assistant_data.dict()
    )
    
    db.add(new_assistant)
    db.commit()
    db.refresh(new_assistant)
    
    return AssistantResponse.from_orm(new_assistant)

@router.get("", response_model=List[AssistantResponse])
async def list_assistants(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lista assistentes do usuário"""
    
    assistants = db.query(Assistant).filter(
        Assistant.user_id == current_user.id
    ).all()
    
    return [AssistantResponse.from_orm(a) for a in assistants]

@router.get("/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(
    assistant_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retorna assistente específico"""
    
    assistant = db.query(Assistant).filter(
        Assistant.id == assistant_id,
        Assistant.user_id == current_user.id
    ).first()
    
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistente não encontrado"
        )
    
    return AssistantResponse.from_orm(assistant)

@router.put("/{assistant_id}", response_model=AssistantResponse)
async def update_assistant(
    assistant_id: int,
    assistant_data: AssistantUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Atualiza assistente"""
    
    assistant = db.query(Assistant).filter(
        Assistant.id == assistant_id,
        Assistant.user_id == current_user.id
    ).first()
    
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistente não encontrado"
        )
    
    # Atualiza campos
    for key, value in assistant_data.dict(exclude_unset=True).items():
        setattr(assistant, key, value)
    
    assistant.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(assistant)
    
    return AssistantResponse.from_orm(assistant)

@router.delete("/{assistant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assistant(
    assistant_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deleta assistente"""
    
    assistant = db.query(Assistant).filter(
        Assistant.id == assistant_id,
        Assistant.user_id == current_user.id
    ).first()
    
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistente não encontrado"
        )
    
    db.delete(assistant)
    db.commit()

@router.post("/{assistant_id}/documents", response_model=DocumentResponse)
async def upload_document(
    assistant_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload de documento para assistente"""
    
    # Verifica se assistente existe e pertence ao usuário
    assistant = db.query(Assistant).filter(
        Assistant.id == assistant_id,
        Assistant.user_id == current_user.id
    ).first()
    
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistente não encontrado"
        )
    
    # Salva arquivo
    upload_dir = Path(f"./data/tenants/{current_user.id}/assistants/{assistant_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Cria registro no banco
    file_size = file_path.stat().st_size
    
    new_document = Document(
        assistant_id=assistant_id,
        filename=file.filename,
        file_path=str(file_path),
        file_type=file.content_type,
        file_size=file_size,
        is_processed=False
    )
    
    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    
    # TODO: Processar documento em background (adicionar à fila)
    
    return DocumentResponse.from_orm(new_document)

@router.get("/{assistant_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    assistant_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lista documentos do assistente"""
    
    # Verifica se assistente pertence ao usuário
    assistant = db.query(Assistant).filter(
        Assistant.id == assistant_id,
        Assistant.user_id == current_user.id
    ).first()
    
    if not assistant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistente não encontrado"
        )
    
    documents = db.query(Document).filter(
        Document.assistant_id == assistant_id
    ).all()
    
    return [DocumentResponse.from_orm(d) for d in documents]
