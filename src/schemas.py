from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# ==================== USER ====================
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    plan: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# ==================== ASSISTANT ====================
class AssistantBase(BaseModel):
    name: str
    description: Optional[str] = None
    llm_provider: str = "groq"
    llm_model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    use_agent: bool = True
    use_cache: bool = True

class AssistantCreate(AssistantBase):
    pass

class AssistantUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    llm_provider: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    use_agent: Optional[bool] = None

class AssistantResponse(AssistantBase):
    id: int
    user_id: int
    total_messages: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# ==================== DOCUMENT ====================
class DocumentResponse(BaseModel):
    id: int
    assistant_id: int
    filename: str
    file_type: Optional[str]
    file_size: int
    is_processed: bool
    chunks_count: int
    uploaded_at: datetime
    
    class Config:
        from_attributes = True

# ==================== CONVERSATION ====================
class ConversationResponse(BaseModel):
    id: int
    assistant_id: int
    user_message: str
    assistant_message: str
    tokens_used: int
    tools_used: Optional[List[str]]
    created_at: datetime
    
    class Config:
        from_attributes = True

# ==================== QUERY ====================
class QueryRequest(BaseModel):
    message: str
    assistant_id: int

class QueryResponse(BaseModel):
    answer: str
    tokens_used: int
    tools_used: List[str]
    conversation_id: int
