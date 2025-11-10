from typing import Dict, Any, Optional
import asyncio
import aiohttp
import os
from pathlib import Path

class WhatsAppBot:
    """Bot WhatsApp integrado com RAG"""
    
    def __init__(self, rag_system, agent_system, evolution_url="http://localhost:8080", api_key="sua_chave_secreta_aqui_123"):
        self.rag_system = rag_system
        self.agent_system = agent_system
        self.evolution_url = evolution_url
        self.api_key = api_key
        self.instance_name = "rag-bot"
        self.user_contexts = {}
        
        print("✅ WhatsApp Bot inicializado")
    
    async def create_instance(self):
        """Cria instância do WhatsApp"""
        url = f"{self.evolution_url}/instance/create"
        
        payload = {
            "instanceName": self.instance_name,
            "token": self.api_key,
            "qrcode": True,
            "integration": "WHATSAPP-BAILEYS"
        }
        
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    data = await response.json()
                    print(f"✅ Instância criada: {self.instance_name}")
                    return data
        except Exception as e:
            print(f"❌ Erro ao criar instância: {e}")
            return None
    
    async def get_qrcode(self):
        """Obtém QR Code para conectar WhatsApp"""
        url = f"{self.evolution_url}/instance/connect/{self.instance_name}"
        
        headers = {"apikey": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
                    
                    if "qrcode" in data:
                        qr = data["qrcode"]["base64"]
                        print("\n📱 QR Code obtido!")
                        
                        # Salva QR code como imagem
                        import base64
                        from PIL import Image
                        import io
                        
                        img_data = base64.b64decode(qr.split(',')[1] if ',' in qr else qr)
                        img = Image.open(io.BytesIO(img_data))
                        img.save("whatsapp_qrcode.png")
                        print("💾 QR Code salvo: whatsapp_qrcode.png")
                        
                    return data
        except Exception as e:
            print(f"❌ Erro ao obter QR code: {e}")
            return None
    
    async def send_message(self, phone: str, message: str):
        """Envia mensagem para um número"""
        url = f"{self.evolution_url}/message/sendText/{self.instance_name}"
        
        payload = {
            "number": phone,
            "text": message
        }
        
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    return await response.json()
        except Exception as e:
            print(f"❌ Erro ao enviar mensagem: {e}")
            return None
    
    def process_text_message(self, phone: str, message: str) -> str:
        """Processa mensagem de texto com RAG/Agent"""
        
        # Comandos especiais
        if message.lower() == "/help":
            return """🤖 *Comandos Disponíveis:*

/help - Este menu
/stats - Estatísticas
/clear - Limpa histórico

Envie qualquer pergunta e eu respondo usando os documentos!"""
        
        if message.lower() == "/stats":
            return f"📊 *Estatísticas:*\nSistema funcionando normalmente!"
        
        if message.lower() == "/clear":
            if phone in self.user_contexts:
                del self.user_contexts[phone]
            return "🗑️ Histórico limpo!"
        
        # Processa com Agent
        try:
            answer = ""
            
            if self.agent_system:
                for chunk in self.agent_system.process_stream(message):
                    if chunk['type'] == 'chunk':
                        answer += chunk['data']
            else:
                for chunk in self.rag_system.query_stream(message):
                    if chunk['type'] == 'chunk':
                        answer += chunk['data']
            
            return answer if answer else "Desculpe, não consegui processar."
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            return f"❌ Erro: {str(e)}"
    
    async def handle_webhook(self, data: Dict[str, Any]) -> Optional[str]:
        """Processa webhook do Evolution API"""
        
        try:
            event = data.get("event")
            
            if event not in ["messages.upsert"]:
                return None
            
            message_data = data.get("data", {})
            phone = message_data.get("key", {}).get("remoteJid", "").replace("@s.whatsapp.net", "")
            message_type = message_data.get("messageType")
            
            print(f"\n📱 Mensagem de {phone} (tipo: {message_type})")
            
            response_text = None
            
            # Processa baseado no tipo
            if message_type == "conversation" or message_type == "extendedTextMessage":
                text = message_data.get("message", {}).get("conversation") or \
                       message_data.get("message", {}).get("extendedTextMessage", {}).get("text")
                
                if text:
                    print(f"💬 Texto: {text}")
                    response_text = self.process_text_message(phone, text)
            
            # Envia resposta
            if response_text:
                await self.send_message(phone, response_text)
                print(f"✅ Resposta enviada")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Erro no webhook: {e}")
            import traceback
            traceback.print_exc()
            return None
