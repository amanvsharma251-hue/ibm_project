"""
app.py
======
Streamlit Web UI for the College Admission Agent (IBM Granite RAG).

Run with:
    cd college_admission_agent
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv("config.env")

# On Streamlit Cloud, load secrets into env vars
try:
    import streamlit as _st
    for _k, _v in _st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from agent import AdmissionChatSession, answer_query
from rag_engine import get_or_build_vector_store

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
APP_TITLE = os.getenv("APP_TITLE", "College Admission AI Assistant")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Force light background on chat bubbles regardless of Streamlit theme */
    .chat-user {
        background: #1565c0 !important;
        border-left: 4px solid #90caf9;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 8px;
        margin: 0.5rem 0;
        color: #ffffff !important;
    }
    .chat-user * { color: #ffffff !important; }

    .chat-assistant {
        background: #1e1e2e !important;
        border-left: 4px solid #0f62fe;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 8px;
        margin: 0.5rem 0;
        color: #e0e0e0 !important;
    }
    .chat-assistant * { color: #e0e0e0 !important; }

    /* Sources */
    .sources-box {
        background: #2d2a1e !important;
        border: 1px solid #f9a825;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-size: 0.82rem;
        color: #ffe082 !important;
        margin-top: 0.3rem;
    }

    /* IBM badge */
    .ibm-badge {
        display: inline-block;
        background: #0f62fe;
        color: white !important;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0d1b4b 0%, #0d47a1 100%);
        color: white !important;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid #1565c0;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white !important; }
    .main-header p  { margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.9; color: white !important; }

    /* Sidebar section */
    .sidebar-section {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State initialisation
# ---------------------------------------------------------------------------
if "chat_session" not in st.session_state:
    st.session_state.chat_session = AdmissionChatSession()

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, sources?}]

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

if "vector_store_error" not in st.session_state:
    st.session_state.vector_store_error = None


# ---------------------------------------------------------------------------
# Lazy vector store initialisation
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_vector_store():
    """Build/load the vector store once and cache it for the session."""
    try:
        store = get_or_build_vector_store()
        return store, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>🎓 {APP_TITLE}</h1>
    <p>Springfield Institute of Technology — Powered by IBM Granite on watsonx.ai</p>
    <span class="ibm-badge">IBM Granite · watsonx.ai · RAG</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.markdown("### ⚙️ Agent Settings")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    model_display = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct")
    st.markdown(f"**LLM Model:** `{model_display}`")
    st.markdown(f"**Embedding:** `{os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}`")
    st.markdown(f"**Top-K Retrieval:** `{os.getenv('RAG_TOP_K', '5')}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")

    kb_path = Path(__file__).parent / "knowledge_base"
    kb_files = list(kb_path.glob("*.txt")) + list(kb_path.glob("*.pdf"))
    if kb_files:
        for f in kb_files:
            st.markdown(f"📄 `{f.name}`")
    else:
        st.warning("No knowledge base files found.")

    st.markdown("---")
    st.markdown("### 🔧 Actions")

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        import shutil
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        if Path(chroma_dir).exists():
            shutil.rmtree(chroma_dir)
        st.cache_resource.clear()
        st.session_state.vector_store_ready = False
        st.success("Knowledge base cleared. Reload the page to rebuild.")
        st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_session.reset()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("""
    **Admissions Office**
    📧 admissions@sit.edu.in
    📞 +91-98765-43210
    🕘 Mon–Fri, 9 AM – 5 PM IST
    """)


# ---------------------------------------------------------------------------
# Vector store status check
# ---------------------------------------------------------------------------
with st.spinner("🔍 Loading knowledge base …"):
    vs, vs_error = init_vector_store()

if vs_error:
    st.error(f"❌ Could not load knowledge base: {vs_error}")
    st.stop()
else:
    st.success("✅ Knowledge base loaded and ready!", icon="📚")


# ---------------------------------------------------------------------------
# Quick Question buttons
# ---------------------------------------------------------------------------
st.markdown("#### 💡 Quick Questions")

quick_questions = [
    "What are the eligibility criteria for B.Tech?",
    "What is the fee for MBA?",
    "What is the last date to apply?",
    "Which entrance exams are accepted?",
    "Are there any scholarships available?",
    "How do I apply for admission?",
]

cols = st.columns(3)
for i, q in enumerate(quick_questions):
    if cols[i % 3].button(q, key=f"qq_{i}", use_container_width=True):
        st.session_state._pending_quick_question = q


# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### 💬 Conversation")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 <strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 <strong>AI Assistant:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                st.markdown(
                    f'<div class="sources-box">📚 <strong>Sources:</strong> {", ".join(msg["sources"])}</div>',
                    unsafe_allow_html=True,
                )

if not st.session_state.messages:
    st.info(
        "👋 Hello! I'm your College Admission AI Assistant for Springfield Institute of Technology.\n\n"
        "Ask me anything about admissions, courses, fees, deadlines, or scholarships!"
    )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask about admissions, courses, fees, deadlines …")

# Handle quick question selection
if hasattr(st.session_state, "_pending_quick_question"):
    user_input = st.session_state._pending_quick_question
    del st.session_state._pending_quick_question


if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("🤔 IBM Granite is thinking …"):
        result = st.session_state.chat_session.chat(user_input)

    # Add assistant message to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })

    # Rerun to display updated messages
    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.8rem; color:#888; padding: 0.5rem 0;">
    🎓 Springfield Institute of Technology &nbsp;|&nbsp;
    Powered by <strong>IBM Granite</strong> on <strong>IBM watsonx.ai</strong> &nbsp;|&nbsp;
    Built with LangChain &amp; Streamlit
</div>
""", unsafe_allow_html=True)
