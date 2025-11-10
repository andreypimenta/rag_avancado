from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from src.database import Base

class User(Base):
    """Usuários da plataforma"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    
    # Plano
    plan = Column(String, default="free")  # free, pro, enterprise
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relações
    assistants = relationship("Assistant", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete-orphan")

class Assistant(Base):
    """Assistentes criados pelos usuários"""
    __tablename__ = "assistants"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Config
    name = Column(String, nullable=False)
    description = Column(Text)
    llm_provider = Column(String, default="groq")
    llm_model = Column(String)
    system_prompt = Column(Text)
    
    # Settings
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2000)
    use_agent = Column(Boolean, default=True)
    use_cache = Column(Boolean, default=True)
    
    # Stats
    total_messages = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relações
    user = relationship("User", back_populates="assistants")
    documents = relationship("Document", back_populates="assistant", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="assistant", cascade="all, delete-orphan")

class Document(Base):
    """Documentos uploadados"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=False)
    
    # File info
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)
    file_size = Column(Integer)
    
    # Processing
    is_processed = Column(Boolean, default=False)
    chunks_count = Column(Integer, default=0)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relações
    assistant = relationship("Assistant", back_populates="documents")

class Conversation(Base):
    """Conversas/mensagens"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    assistant_id = Column(Integer, ForeignKey("assistants.id"), nullable=False)
    
    # Message
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=False)
    
    # Metadata
    tokens_used = Column(Integer, default=0)
    latency_ms = Column(Integer)
    tools_used = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relações
    assistant = relationship("Assistant", back_populates="conversations")

class APIKey(Base):
    """API Keys para acesso programático"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Key
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relações
    user = relationship("User", back_populates="api_keys")

class UsageLog(Base):
    """Log de uso para billing"""
    __tablename__ = "usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assistant_id = Column(Integer, ForeignKey("assistants.id"))
    
    # Usage
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    operation = Column(String)  # query, upload, etc
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relações
    user = relationship("User", back_populates="usage_logs")
