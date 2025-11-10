import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="RAG Universal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTextInput > div > div > input { font-size: 16px; }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user { background-color: #2b313e; }
    .chat-message.assistant { background-color: #1e1e1e; }
    .chat-message .message {
        color: #fff;
        font-size: 16px;
        line-height: 1.6;
    }
    .chat-message .metadata {
        color: #888;
        font-size: 12px;
        margin-top: 0.5rem;
    }
    .reasoning-box {
        background-color: #1a1a2e;
        border-left: 3px solid #4a9eff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 RAG Universal")
st.markdown("### Sistema de Recuperação e Geração com IA")

API_URL = "http://localhost:8000"

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # ==================== MODO AGENT ====================
    st.subheader("🤖 Modo Agent")
    use_agent = st.checkbox(
        "Agent Autônomo", 
        value=False,
        help="IA analisa, planeja e escolhe ferramentas automaticamente"
    )
    
    if use_agent:
        st.info("🧠 Agent ativo! Vai decidir sozinho quais ferramentas usar")
    
    st.divider()
    
    # ==================== RETRIEVAL ====================
    st.subheader("🔍 Retrieval")
    use_hybrid = st.checkbox("Hybrid Search", value=True, disabled=use_agent)
    use_rerank = st.checkbox("Reranking", value=True, disabled=use_agent)
    use_expansion = st.checkbox("Query Expansion", value=False, disabled=use_agent)
    
    if use_agent:
        st.caption("⚠️ Desabilitado no modo Agent")
    
    st.divider()
    
    # ==================== UPLOAD ====================
    st.subheader("📁 Upload de Documentos")
    uploaded_file = st.file_uploader(
        "Adicionar arquivo ao RAG",
        type=['pdf', 'txt', 'docx', 'xlsx', 'csv', 'json', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a']
    )
    
    if uploaded_file is not None:
        if st.button("📤 Processar Arquivo", use_container_width=True):
            with st.spinner("Processando..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data['message']}")
                        st.rerun()
                    else:
                        st.error(f"❌ Erro: {response.text}")
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
    
    st.divider()
    
    # ==================== ESTATÍSTICAS ====================
    st.subheader("📊 Estatísticas")
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            if stats['status'] == 'ok':
                st.metric("Documentos", stats['total_documents'])
                if stats.get('agent_available'):
                    st.success("🤖 Agent disponível")
    except:
        st.error("API não conectada")
    
    st.divider()
    
    # ==================== CACHE ====================
    st.subheader("💾 Cache & Economia")
    try:
        response = requests.get(f"{API_URL}/cache/stats")
        if response.status_code == 200:
            cache = response.json()
            
            if cache['status'] == 'ok':
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Hit Rate", 
                        f"{cache.get('response_hit_rate', 0):.1f}%"
                    )
                    st.metric(
                        "Economia",
                        f"${cache.get('estimated_cost_saved', 0):.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Tokens Saved",
                        f"{cache.get('total_tokens_saved', 0):,}"
                    )
                    st.metric(
                        "Cache Size",
                        f"{cache.get('total_mb', 0):.1f} MB"
                    )
                
                if st.button("🗑️ Limpar Cache", use_container_width=True):
                    try:
                        clear = requests.post(f"{API_URL}/cache/clear")
                        if clear.status_code == 200:
                            st.success("Cache limpo!")
                            st.rerun()
                    except:
                        st.error("Erro ao limpar")
    except:
        st.info("Cache indisponível")

# ==================== HISTÓRICO DE CHAT ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message">👤 <strong>Você:</strong><br>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = message.get("confidence", 0)
            method = message.get("method", "")
            agent_mode = message.get("agent_mode", False)
            tools_used = message.get("tools_used", [])
            
            confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
            
            # Badge do Agent
            agent_badge = "🤖 <strong>AGENT</strong> | " if agent_mode else ""
            tools_badge = f"Tools: {', '.join(tools_used)} | " if tools_used else ""
            
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message">🤖 <strong>Assistente:</strong><br>{message["content"]}</div>
                <div class="metadata">{agent_badge}{tools_badge}{confidence_color} Confiança: {confidence:.1%} | Método: {method}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Exibe fontes
            if "sources" in message and message["sources"]:
                with st.expander("📚 Ver Fontes"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        score = source.get('score', 0)
                        score_emoji = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                        st.markdown(f"{score_emoji} **[{i}]** {source['metadata'].get('source', 'Unknown')} (score: {score:.2f})")
                        st.text(source['content'])

# ==================== INPUT ====================
if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message">👤 <strong>Você:</strong><br>{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mensagem de status baseada no modo
    if use_agent:
        spinner_msg = "🧠 Agent analisando e decidindo ações..."
    else:
        spinner_msg = "🔍 Buscando e gerando resposta..."
    
    with st.spinner(spinner_msg):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": prompt,
                    "use_hybrid": use_hybrid,
                    "use_rerank": use_rerank,
                    "use_expansion": use_expansion,
                    "use_agent": use_agent  # NOVO!
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "confidence": data.get("confidence", 0),
                    "method": data.get("method", ""),
                    "sources": data.get("sources", []),
                    "agent_mode": data.get("agent_mode", False),
                    "tools_used": data.get("tools_used", [])
                })
                
                st.rerun()
            else:
                st.error(f"❌ Erro: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            st.info("💡 Certifique-se de que a API está rodando")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    RAG Universal v2.0 | Agent Mode + Cache + Multi-Modal | FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
