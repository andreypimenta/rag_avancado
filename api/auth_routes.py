from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.database import get_db
from src.database.models import User
from src.auth import get_password_hash, authenticate_user, create_access_token, get_current_user
from src.schemas import UserCreate, UserLogin, Token, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """Cadastro de novo usuário"""
    
    # Verifica se email já existe
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email já cadastrado"
        )
    
    # Cria usuário
    hashed_password = get_password_hash(user_data.password)
    
    new_user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        plan="free"
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Gera token
    access_token = create_access_token(data={"sub": new_user.id})
    
    return Token(
        access_token=access_token,
        user=UserResponse.from_orm(new_user)
    )

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login de usuário"""
    
    user = authenticate_user(db, credentials.email, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou senha incorretos"
        )
    
    # Gera token
    access_token = create_access_token(data={"sub": user.id})
    
    return Token(
        access_token=access_token,
        user=UserResponse.from_orm(user)
    )

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Retorna usuário atual"""
    return UserResponse.from_orm(current_user)
