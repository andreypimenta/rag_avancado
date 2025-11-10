import hashlib
import json
from pathlib import Path
from typing import Optional, Any, List
from datetime import datetime, timedelta
import pickle
from diskcache import Cache

class CacheManager:
    """Gerenciador de cache para otimização de custos"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.embedding_cache = Cache(str(self.cache_dir / "embeddings"))
        self.response_cache = Cache(str(self.cache_dir / "responses"))
        self.stats_cache = Cache(str(self.cache_dir / "stats"))
        
        if not self.stats_cache.get('initialized'):
            self._init_stats()
        
        print("✅ Cache Manager inicializado")
    
    def _init_stats(self):
        """Inicializa estatísticas"""
        stats = {
            'initialized': True,
            'embedding_hits': 0,
            'embedding_misses': 0,
            'response_hits': 0,
            'response_misses': 0,
            'total_tokens_saved': 0,
            'estimated_cost_saved': 0.0,
            'last_reset': datetime.now().isoformat()
        }
        for key, value in stats.items():
            self.stats_cache.set(key, value)
    
    def _hash_content(self, content: str) -> str:
        """Gera hash do conteúdo"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_response_cache(self, question: str) -> Optional[dict]:
        """Busca resposta em cache"""
        question_hash = self._hash_content(question.lower().strip())
        cached = self.response_cache.get(question_hash)
        
        if cached:
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=24):
                hits = self.stats_cache.get('response_hits', 0)
                self.stats_cache.set('response_hits', hits + 1)
                
                saved_cost = self.stats_cache.get('estimated_cost_saved', 0.0)
                saved_tokens = self.stats_cache.get('total_tokens_saved', 0)
                self.stats_cache.set('estimated_cost_saved', saved_cost + 0.01)
                self.stats_cache.set('total_tokens_saved', saved_tokens + cached.get('tokens', 500))
                
                print(f"   💾 Cache HIT - Resposta")
                return cached['response']
        
        misses = self.stats_cache.get('response_misses', 0)
        self.stats_cache.set('response_misses', misses + 1)
        return None
    
    def set_response_cache(self, question: str, response: dict, tokens_used: int = 500):
        """Salva resposta em cache"""
        question_hash = self._hash_content(question.lower().strip())
        
        cache_data = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'tokens': tokens_used
        }
        
        self.response_cache.set(question_hash, cache_data, expire=86400)
    
    def get_stats(self) -> dict:
        """Retorna estatísticas"""
        stats = {}
        for key in ['embedding_hits', 'embedding_misses', 'response_hits', 
                    'response_misses', 'total_tokens_saved', 'estimated_cost_saved',
                    'last_reset']:
            stats[key] = self.stats_cache.get(key, 0)
        
        total_embedding = stats['embedding_hits'] + stats['embedding_misses']
        total_response = stats['response_hits'] + stats['response_misses']
        
        stats['embedding_hit_rate'] = (
            stats['embedding_hits'] / total_embedding * 100 
            if total_embedding > 0 else 0
        )
        
        stats['response_hit_rate'] = (
            stats['response_hits'] / total_response * 100 
            if total_response > 0 else 0
        )
        
        return stats
    
    def clear_cache(self, cache_type: str = 'all'):
        """Limpa cache"""
        if cache_type in ['all', 'responses']:
            self.response_cache.clear()
            print("🗑️  Cache limpo")
        
        if cache_type == 'all':
            self._init_stats()
    
    def get_cache_size(self) -> dict:
        """Retorna tamanho do cache"""
        def get_dir_size(path):
            total = 0
            try:
                for entry in Path(path).rglob('*'):
                    if entry.is_file():
                        total += entry.stat().st_size
            except:
                pass
            return total / (1024 * 1024)
        
        return {
            'total_mb': round(get_dir_size(self.cache_dir), 2)
        }
