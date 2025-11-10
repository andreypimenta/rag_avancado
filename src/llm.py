from abc import ABC, abstractmethod
from typing import Optional, Iterator
import os

class BaseLLM(ABC):
    """Interface base para qualquer LLM"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """Gera resposta com streaming"""
        pass

class ClaudeLLM(BaseLLM):
    """Implementação para Claude (Anthropic)"""
    
    def __init__(self, model="claude-sonnet-4-20250514"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt or "Você é um assistente útil.",
            messages=messages
        )
        return response.content[0].text
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = [{"role": "user", "content": prompt}]
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=system_prompt or "Você é um assistente útil.",
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text

class OpenAILLM(BaseLLM):
    """Implementação para GPT"""
    
    def __init__(self, model="gpt-4"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system_prompt or "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class GroqLLM(BaseLLM):
    """Implementação para Groq"""
    
    def __init__(self, model="llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system_prompt or "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class GeminiLLM(BaseLLM):
    """Implementação para Google Gemini"""
    
    def __init__(self, model="gemini-pro"):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(full_prompt)
        return response.text
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(full_prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text

class DeepSeekLLM(BaseLLM):
    """Implementação para DeepSeek"""
    
    def __init__(self, model="deepseek-chat"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system_prompt or "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class LLMFactory:
    @staticmethod
    def create(provider: str = "groq", **kwargs):
        providers = {
            "claude": ClaudeLLM,
            "openai": OpenAILLM,
            "groq": GroqLLM,
            "gemini": GeminiLLM,
            "deepseek": DeepSeekLLM,
        }
        
        if provider not in providers:
            raise ValueError(f"Provider não suportado: {provider}")
        
        return providers[provider](**kwargs)
