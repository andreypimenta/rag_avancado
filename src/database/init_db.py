from src.database import engine, Base
from src.database.models import User, Assistant, Document, Conversation, APIKey, UsageLog

def init_database():
    """Cria todas as tabelas"""
    print("🗄️  Criando tabelas no banco de dados...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tabelas criadas com sucesso!")

if __name__ == "__main__":
    init_database()
