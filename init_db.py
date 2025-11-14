"""
Inicializa o banco de dados criando todas as tabelas
"""
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent))

try:
    from src.database import engine, Base
    from src.database.models import User
    
    def init_database():
        """Cria todas as tabelas no banco"""
        print("🔧 Inicializando banco de dados...")
        print(f"📍 Engine: {engine.url}")
        
        try:
            # Cria todas as tabelas
            Base.metadata.create_all(bind=engine)
            print("✅ Tabelas criadas com sucesso!")
            return True
        except Exception as e:
            print(f"⚠️  Aviso ao criar tabelas: {e}")
            return False
    
    if __name__ == "__main__":
        success = init_database()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"⚠️  Módulos de database não encontrados: {e}")
    print("   (Ignorando - talvez não esteja usando multi-tenant)")
    sys.exit(0)
except Exception as e:
    print(f"⚠️  Erro na inicialização: {e}")
    print("   (Continuando mesmo assim...)")
    sys.exit(0)
