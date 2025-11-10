from typing import List, Dict, Any, Optional
from pathlib import Path
import pytesseract
from PIL import Image
import pandas as pd
import json
import pdfplumber
from langchain_core.documents import Document
import cv2
import numpy as np

class MultiModalProcessor:
    """Processa múltiplos tipos de arquivos incluindo áudio, vídeo e OCR avançado"""
    
    def __init__(self, enable_whisper: bool = True, enable_video: bool = True):
        self.supported_extensions = {
            # Texto
            '.txt', '.md', '.csv', '.json',
            # Documentos
            '.pdf', '.docx', '.doc',
            # Planilhas
            '.xlsx', '.xls',
            # Imagens
            '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp',
            # Áudio
            '.mp3', '.wav', '.m4a', '.ogg', '.flac',
            # Vídeo
            '.mp4', '.avi', '.mov', '.mkv'
        }
        
        self.enable_whisper = enable_whisper
        self.enable_video = enable_video
        
        # Tenta importar whisper
        if self.enable_whisper:
            try:
                import whisper
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper carregado (modelo base)")
            except:
                print("⚠️  Whisper não disponível")
                self.enable_whisper = False
        
        print("✅ Processador Multi-modal Plus inicializado")
    
    def process_file(self, file_path: str) -> List[Document]:
        """Processa arquivo e retorna documentos"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        print(f"📄 Processando: {path.name} ({extension})")
        
        try:
            # Imagens
            if extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
                return self._process_image_advanced(file_path)
            
            # PDFs
            elif extension == '.pdf':
                return self._process_pdf_with_ocr(file_path)
            
            # Planilhas
            elif extension in ['.xlsx', '.xls']:
                return self._process_excel(file_path)
            
            # CSV
            elif extension == '.csv':
                return self._process_csv(file_path)
            
            # JSON
            elif extension == '.json':
                return self._process_json(file_path)
            
            # Texto
            elif extension == '.txt':
                return self._process_text(file_path)
            
            # Áudio
            elif extension in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
                if self.enable_whisper:
                    return self._process_audio(file_path)
                else:
                    print(f"⚠️  Áudio não suportado (Whisper não disponível)")
                    return []
            
            # Vídeo
            elif extension in ['.mp4', '.avi', '.mov', '.mkv']:
                if self.enable_video:
                    return self._process_video(file_path)
                else:
                    print(f"⚠️  Vídeo não suportado")
                    return []
            
            else:
                print(f"⚠️  Tipo não suportado: {extension}")
                return []
                
        except Exception as e:
            print(f"❌ Erro ao processar {path.name}: {e}")
            return []
    
    def _process_image_advanced(self, file_path: str) -> List[Document]:
        """Processa imagem com OCR melhorado e análise"""
        try:
            image = Image.open(file_path)
            documents = []
            
            # 1. Detecta se é gráfico/diagrama
            is_chart = self._detect_chart(image)
            
            if is_chart:
                print(f"   📊 Gráfico detectado!")
            
            # 2. OCR padrão (Tesseract)
            text = pytesseract.image_to_string(image, lang='por+eng')
            
            if text.strip():
                content = text
                
                # Adiciona descrição se for gráfico
                if is_chart:
                    content = f"[GRÁFICO/DIAGRAMA DETECTADO]\n\n{content}"
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        'source': Path(file_path).name,
                        'file_type': Path(file_path).suffix,
                        'processing': 'ocr_advanced',
                        'is_chart': is_chart,
                        'text_length': len(text)
                    }
                ))
                
                print(f"   ✅ OCR extraiu {len(text)} caracteres")
            else:
                print(f"   ⚠️  Nenhum texto encontrado")
            
            # 3. Metadados da imagem
            img_info = f"Imagem: {image.size[0]}x{image.size[1]}px, Modo: {image.mode}"
            documents.append(Document(
                page_content=img_info,
                metadata={
                    'source': Path(file_path).name,
                    'file_type': Path(file_path).suffix,
                    'processing': 'image_metadata'
                }
            ))
            
            return documents
            
        except Exception as e:
            print(f"   ❌ Erro no OCR: {e}")
            return []
    
    def _detect_chart(self, image: Image.Image) -> bool:
        """Detecta se imagem contém gráfico/diagrama"""
        try:
            # Converte para OpenCV
            img_array = np.array(image)
            
            # Se colorida, converte para grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detecta linhas (gráficos geralmente têm muitas linhas)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
            
            # Se tem muitas linhas, provavelmente é gráfico
            if lines is not None and len(lines) > 10:
                return True
            
            return False
            
        except:
            return False
    
    def _process_audio(self, file_path: str) -> List[Document]:
        """Transcreve áudio usando Whisper"""
        try:
            import whisper
            
            print(f"   🎤 Transcrevendo áudio...")
            
            # Transcreve
            result = self.whisper_model.transcribe(
                file_path,
                language='pt',  # Pode detectar automaticamente também
                fp16=False
            )
            
            text = result['text']
            language = result.get('language', 'unknown')
            
            print(f"   ✅ Transcrição completa ({len(text)} caracteres, idioma: {language})")
            
            return [Document(
                page_content=text,
                metadata={
                    'source': Path(file_path).name,
                    'file_type': Path(file_path).suffix,
                    'processing': 'audio_transcription',
                    'language': language,
                    'duration': result.get('duration', 0)
                }
            )]
            
        except Exception as e:
            print(f"   ❌ Erro na transcrição: {e}")
            return []
    
    def _process_video(self, file_path: str) -> List[Document]:
        """Processa vídeo: extrai frames + áudio"""
        try:
            from moviepy.editor import VideoFileClip
            import tempfile
            
            documents = []
            
            print(f"   🎬 Processando vídeo...")
            
            # Carrega vídeo
            clip = VideoFileClip(file_path)
            duration = clip.duration
            fps = clip.fps
            
            print(f"   ⏱️  Duração: {duration:.1f}s, FPS: {fps:.1f}")
            
            # 1. Extrai áudio e transcreve
            if self.enable_whisper and clip.audio is not None:
                print(f"   🎤 Extraindo áudio...")
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    clip.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                    
                    # Transcreve
                    audio_docs = self._process_audio(temp_audio.name)
                    documents.extend(audio_docs)
                    
                    # Remove arquivo temporário
                    Path(temp_audio.name).unlink()
            
            # 2. Extrai frames-chave (1 por segundo)
            print(f"   📸 Extraindo frames...")
            
            frame_interval = max(1, int(duration / 10))  # Máximo 10 frames
            
            for i in range(0, int(duration), frame_interval):
                try:
                    frame = clip.get_frame(i)
                    
                    # Converte para PIL Image
                    img = Image.fromarray(frame)
                    
                    # OCR no frame
                    text = pytesseract.image_to_string(img, lang='por+eng')
                    
                    if text.strip() and len(text) > 20:
                        documents.append(Document(
                            page_content=f"Frame em {i}s:\n{text}",
                            metadata={
                                'source': Path(file_path).name,
                                'file_type': Path(file_path).suffix,
                                'processing': 'video_frame_ocr',
                                'timestamp': i
                            }
                        ))
                except:
                    pass
            
            # 3. Metadados do vídeo
            video_info = f"Vídeo: {duration:.1f}s, {clip.size[0]}x{clip.size[1]}px, {fps:.1f} fps"
            documents.append(Document(
                page_content=video_info,
                metadata={
                    'source': Path(file_path).name,
                    'file_type': Path(file_path).suffix,
                    'processing': 'video_metadata'
                }
            ))
            
            clip.close()
            
            print(f"   ✅ Vídeo processado: {len(documents)} elementos extraídos")
            return documents
            
        except Exception as e:
            print(f"   ❌ Erro ao processar vídeo: {e}")
            return []
    
    def _process_pdf_with_ocr(self, file_path: str) -> List[Document]:
        """Processa PDF com texto e OCR"""
        documents = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"   📖 {total_pages} páginas")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Texto
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                'source': Path(file_path).name,
                                'page': page_num,
                                'file_type': '.pdf',
                                'processing': 'text_extraction'
                            }
                        ))
                    else:
                        # OCR
                        print(f"   🔍 Página {page_num} - OCR...")
                        img = page.to_image(resolution=300)
                        pil_img = img.original
                        ocr_text = pytesseract.image_to_string(pil_img, lang='por+eng')
                        
                        if ocr_text.strip():
                            documents.append(Document(
                                page_content=ocr_text,
                                metadata={
                                    'source': Path(file_path).name,
                                    'page': page_num,
                                    'file_type': '.pdf',
                                    'processing': 'ocr'
                                }
                            ))
                    
                    # Tabelas
                    tables = page.extract_tables()
                    if tables:
                        for i, table in enumerate(tables, 1):
                            table_text = self._table_to_text(table)
                            documents.append(Document(
                                page_content=f"Tabela {i}:\n{table_text}",
                                metadata={
                                    'source': Path(file_path).name,
                                    'page': page_num,
                                    'table': i,
                                    'file_type': '.pdf',
                                    'processing': 'table_extraction'
                                }
                            ))
            
            print(f"   ✅ {len(documents)} elementos extraídos")
            return documents
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return []
    
    def _process_excel(self, file_path: str) -> List[Document]:
        """Processa Excel"""
        documents = []
        
        try:
            xl_file = pd.ExcelFile(file_path)
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content = f"Planilha: {sheet_name}\n\n{df.to_string(index=False)}"
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        'source': Path(file_path).name,
                        'sheet': sheet_name,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'file_type': Path(file_path).suffix,
                        'processing': 'excel'
                    }
                ))
            
            print(f"   ✅ {len(documents)} planilhas")
            return documents
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return []
    
    def _process_csv(self, file_path: str) -> List[Document]:
        """Processa CSV"""
        try:
            df = pd.read_csv(file_path)
            content = df.to_string(index=False)
            
            return [Document(
                page_content=content,
                metadata={
                    'source': Path(file_path).name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'file_type': '.csv',
                    'processing': 'csv'
                }
            )]
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return []
    
    def _process_json(self, file_path: str) -> List[Document]:
        """Processa JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            
            return [Document(
                page_content=content,
                metadata={
                    'source': Path(file_path).name,
                    'file_type': '.json',
                    'processing': 'json'
                }
            )]
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return []
    
    def _process_text(self, file_path: str) -> List[Document]:
        """Processa texto"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return [Document(
                page_content=content,
                metadata={
                    'source': Path(file_path).name,
                    'file_type': '.txt',
                    'processing': 'text'
                }
            )]
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return []
    
    def _table_to_text(self, table: List[List]) -> str:
        """Converte tabela para texto"""
        if not table:
            return ""
        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
        return df.to_string(index=False)
