import os
# Fix protobuf compiler descriptor compatibility issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Suppress transformers/tokenizers verbose startup warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import streamlit as st
from dotenv import load_dotenv

import utils
import pdf_processor
import vector_store
import chatbot

# Page configuration — MUST be the first Streamlit command
st.set_page_config(
    page_title="Jan Sahayak AI — Public Policy Gazette",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env
load_dotenv()

# App directories
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"

# Load API key from secrets, environment, or user override
API_KEY = ""

# 1. Try loading from streamlit secrets
try:
    if "GOOGLE_API_KEY" in st.secrets:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
    elif "google_api_key" in st.secrets:
        API_KEY = st.secrets["google_api_key"]
except Exception:
    pass

# 2. Try loading from environment variables
if not API_KEY:
    API_KEY = os.getenv("GOOGLE_API_KEY", "")

if API_KEY == "your_api_key_here":
    API_KEY = ""

# Initialize API key override in session state if not already set
if "api_key_override" not in st.session_state:
    st.session_state.api_key_override = ""

# 3. Use user override if provided
if not API_KEY and st.session_state.api_key_override:
    API_KEY = st.session_state.api_key_override

# Set environment for underlying langchain libraries
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

# ===================== DEMO SCHEMES DATA =====================
DEMO_SCHEMES = [
    {
        "name": "Pradhan Mantri Jan Dhan Yojana",
        "short": "PMJDY",
        "code": "REF. NO. 001 / FINANCIAL INCLUSION",
        "description": "National Mission for Financial Inclusion providing zero-balance bank accounts, RuPay debit cards, accident insurance, and overdraft facilities to unbanked households.",
        "questions": [
            "What is PMJDY?",
            "What is the overdraft limit under PMJDY?",
            "Who is eligible for PMJDY?"
        ]
    },
    {
        "name": "PM Kisan Samman Nidhi",
        "short": "PM-KISAN",
        "code": "REF. NO. 002 / AGRICULTURAL SUPPORT",
        "description": "Direct income support initiative providing ₹6,000 annually in three equal installments directly to small and marginal farmer families across India.",
        "questions": [
            "Who can apply for PM Kisan?",
            "What benefits are provided under PM Kisan?",
            "How much financial assistance is given under PM Kisan?"
        ]
    },
    {
        "name": "Ayushman Bharat (PM-JAY)",
        "short": "PM-JAY",
        "code": "REF. NO. 003 / NATIONAL HEALTH PROTECTION",
        "description": "World's largest government-funded healthcare scheme, providing secondary and tertiary hospitalization coverage up to ₹5 Lakh per family annually.",
        "questions": [
            "What is Ayushman Bharat?",
            "Who can avail the Ayushman Bharat scheme?",
            "What health coverage is provided under Ayushman Bharat?"
        ]
    }
]

# ===================== TUTORIAL VIDEOS =====================
TUTORIAL_VIDEOS = [
    {
        "title": "Document Indexing Protocol",
        "description": "Systematic walkthrough for uploading official government policy PDFs, extracting document text, and vectorizing chunks into the local database.",
        "steps": [
            "Select scheme PDFs via the Editorial Control Panel",
            "Execute 'Process & Index Documents' command",
            "Verify chunk creation and database count status",
            "Initiate grounded semantic search queries"
        ]
    },
    {
        "title": "Query Formulation & Citation Verification",
        "description": "Guidelines for formulating natural language inquiries and auditing returned source citations for fact verification.",
        "steps": [
            "Submit inquiry via the Central Desk input field",
            "Review grounded answer compiled by Gemini 3.6",
            "Inspect 'Verified Source Footnotes' drawer",
            "Cross-check page numbers and confidence scores"
        ]
    }
]

# ===================== MINIMALIST PREMIUM NEWSPAPER CSS =====================
st.markdown("""
<style>
    /* ===== Google Typography ===== */
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Playfair+Display:ital,wght@0,400..700;1,400..700&family=Source+Sans+3:ital,wght@0,300..700;1,300..700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ===== Color Palette & Variables ===== */
    :root {
        --bg-paper: #F7F5F0;
        --surface-white: #FFFFFF;
        --surface-warm: #EFECE6;
        --ink-black: #111111;
        --ink-dark: #222222;
        --text-muted: #5F5B55;
        --text-subtle: #8C867C;
        --border-rule: #D8D4CC;
        --border-dark: #111111;
        --border-heavy: #BDB8AE;
    }

    /* ===== Global Resets & Typography ===== */
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--ink-black) !important;
        background-color: var(--bg-paper) !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: var(--bg-paper) !important;
    }

    /* Editorial Containers */
    .block-container {
        max-width: 1280px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* ===== Newspaper Masthead ===== */
    .masthead-container {
        text-align: center;
        margin-bottom: 1.8rem;
        border-bottom: 2px solid var(--ink-black);
        padding-bottom: 1rem;
    }
    .masthead-meta-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border-rule);
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
    }
    .masthead-title {
        font-family: 'Newsreader', 'Playfair Display', Georgia, serif;
        font-size: 3.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--ink-black);
        margin: 0;
        line-height: 1.05;
        text-transform: uppercase;
    }
    .masthead-subtitle {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.1rem;
        font-style: italic;
        color: var(--text-muted);
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }
    .masthead-nav-bar {
        border-top: 1px solid var(--ink-black);
        border-bottom: 1px solid var(--ink-black);
        padding: 0.35rem 0;
        margin-top: 0.8rem;
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
        color: var(--ink-dark);
        display: flex;
        justify-content: space-around;
    }

    /* ===== Sidebar (Editorial Desk Panel) ===== */
    section[data-testid="stSidebar"] {
        background-color: var(--surface-warm) !important;
        border-right: 1px solid var(--border-rule) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink-black) !important;
    }
    .sidebar-editorial-head {
        font-family: 'Newsreader', serif;
        font-size: 1.4rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }
    .sidebar-editorial-sub {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted) !important;
        margin-bottom: 1rem;
    }

    /* Editorial Section Headers */
    .section-label {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--ink-black) !important;
        border-bottom: 1px solid var(--border-rule);
        padding-bottom: 0.3rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* System Status Table Box */
    .status-box {
        background: var(--surface-white);
        border: 1px solid var(--border-rule);
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #F0ECE6;
    }
    .status-row:last-child {
        border-bottom: none;
    }
    .status-key {
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 1px;
    }
    .status-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--ink-black);
    }
    .status-ready {
        color: #15803d !important;
        font-weight: 600;
    }
    .status-empty {
        color: #b91c1c !important;
        font-weight: 600;
    }

    /* ===== Custom Buttons (Restrained Editorial) ===== */
    .stButton > button {
        border-radius: 2px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: none !important;
    }
    
    /* Primary Black Button */
    .stButton > button[kind="primary"] {
        background-color: var(--ink-black) !important;
        color: var(--surface-white) !important;
        border: 1px solid var(--ink-black) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #333333 !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }

    /* Secondary Bordered Button */
    .stButton > button[kind="secondary"] {
        background-color: var(--surface-white) !important;
        color: var(--ink-black) !important;
        border: 1px solid var(--border-rule) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--surface-warm) !important;
        border-color: var(--ink-black) !important;
        color: var(--ink-black) !important;
    }

    /* ===== Form Controls & Inputs ===== */
    .stTextInput input, .stSelectbox select {
        border-radius: 2px !important;
        border: 1px solid var(--border-rule) !important;
        background-color: var(--surface-white) !important;
        color: var(--ink-black) !important;
        font-family: 'Source Sans 3', sans-serif !important;
    }
    .stTextInput input:focus {
        border-color: var(--ink-black) !important;
        box-shadow: none !important;
    }

    /* ===== Streamlit Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        border-bottom: 2px solid var(--border-rule) !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.8rem !important;
        color: var(--text-muted) !important;
        background-color: transparent !important;
        border-radius: 0px !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink-black) !important;
        border-bottom: 3px solid var(--ink-black) !important;
        background-color: transparent !important;
    }

    /* ===== Front-Page Editorial Frontpiece ===== */
    .editorial-lead-box {
        background: var(--surface-white);
        border: 1px solid var(--border-rule);
        border-top: 3px solid var(--ink-black);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .editorial-headline {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.15;
        color: var(--ink-black);
        margin-bottom: 0.8rem;
    }
    .editorial-subhead {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-muted);
        margin-bottom: 1.5rem;
    }
    .editorial-columns-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        border-top: 1px solid var(--border-rule);
        padding-top: 1.5rem;
    }
    .editorial-col {
        border-right: 1px solid var(--border-rule);
        padding-right: 1.2rem;
    }
    .editorial-col:last-child {
        border-right: none;
        padding-right: 0;
    }
    .editorial-col-num {
        font-family: 'Newsreader', serif;
        font-size: 1.1rem;
        font-weight: 700;
        font-style: italic;
        color: var(--ink-black);
        margin-bottom: 0.3rem;
    }
    .editorial-col-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--ink-black);
        margin-bottom: 0.4rem;
    }
    .editorial-col-body {
        font-size: 0.88rem;
        line-height: 1.55;
        color: var(--text-muted);
    }

    /* ===== Scheme Editorial Column Cards ===== */
    .scheme-editorial-card {
        background: var(--surface-white);
        border: 1px solid var(--border-rule);
        border-top: 3px solid var(--ink-black);
        padding: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .scheme-editorial-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-subtle);
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .scheme-editorial-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--ink-black);
        margin-bottom: 0.6rem;
        line-height: 1.2;
    }
    .scheme-editorial-body {
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--text-muted);
        margin-bottom: 1.2rem;
    }

    /* ===== Chat Messages ===== */
    .stChatMessage {
        background-color: var(--surface-white) !important;
        border: 1px solid var(--border-rule) !important;
        border-radius: 2px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
    }

    /* ===== Source Citation Footnotes ===== */
    .footnote-box {
        background-color: var(--bg-paper);
        border: 1px solid var(--border-rule);
        border-left: 3px solid var(--ink-black);
        padding: 0.9rem 1.1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }
    .footnote-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--ink-black);
        border-bottom: 1px solid var(--border-rule);
        padding-bottom: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .footnote-badge {
        background: var(--ink-black);
        color: var(--surface-white);
        padding: 2px 8px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .footnote-body {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.85rem;
        line-height: 1.6;
        color: var(--text-muted);
        white-space: pre-wrap;
    }

    /* ===== Chat Input Container ===== */
    div[data-testid="stChatInput"] {
        border-radius: 2px !important;
        border: 1px solid var(--ink-black) !important;
        background: var(--surface-white) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-family: 'Source Sans 3', sans-serif !important;
        color: var(--ink-black) !important;
    }
    .stBottom, div[data-testid="stBottom"] {
        background-color: var(--bg-paper) !important;
        border-top: 1px solid var(--border-rule) !important;
    }

    /* ===== Expander Customization ===== */
    .streamlit-expanderHeader {
        background: var(--surface-warm) !important;
        border: 1px solid var(--border-rule) !important;
        border-radius: 2px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: var(--ink-black) !important;
    }

    /* ===== Dividers ===== */
    hr {
        border-color: var(--border-rule) !important;
        margin: 1.5rem 0 !important;
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-paper); }
    ::-webkit-scrollbar-thumb { background: var(--border-heavy); }
    ::-webkit-scrollbar-thumb:hover { background: var(--ink-black); }

    /* Warning / Lock Notice */
    .editorial-notice {
        background: var(--surface-white);
        border: 1px solid var(--border-heavy);
        border-left: 4px solid var(--ink-black);
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: var(--ink-black);
    }
</style>
""", unsafe_allow_html=True)

# Ensure folders exist
utils.ensure_directories([UPLOAD_DIR, CHROMA_DIR])

# ===================== SESSION STATE =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    if os.path.exists(UPLOAD_DIR):
        files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.pdf')]
        st.session_state.uploaded_files = files

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}

# Lazy load vector database if chroma.sqlite3 exists
if st.session_state.vector_db is None:
    chroma_sqlite = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if os.path.exists(chroma_sqlite) and len(st.session_state.uploaded_files) > 0:
        try:
            st.session_state.vector_db = vector_store.get_vector_store(CHROMA_DIR, API_KEY)
        except Exception:
            pass


# ===================== SIDEBAR (Editorial Control Panel) =====================
with st.sidebar:
    st.markdown("""
    <div style="padding-top: 0.5rem;">
        <div class="sidebar-editorial-head">Editorial Desk</div>
        <div class="sidebar-editorial-sub">Document Control & System Registry</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---- System Status Box ----
    is_connected = bool(API_KEY)
    status_text = "CONNECTED" if is_connected else "DISCONNECTED"
    status_class = "status-ready" if is_connected else "status-empty"
    
    chunk_count = 0
    if st.session_state.vector_db is not None:
        chunk_count = vector_store.get_chunk_count(st.session_state.vector_db)
    num_files = len(st.session_state.uploaded_files)
    
    st.markdown(f"""
    <div class="section-label">System Audit</div>
    <div class="status-box">
        <div class="status-row">
            <span class="status-key">AI Service</span>
            <span class="status-val {status_class}">{status_text}</span>
        </div>
        <div class="status-row">
            <span class="status-key">LLM Model</span>
            <span class="status-val">Gemini 3.6 Flash</span>
        </div>
        <div class="status-row">
            <span class="status-key">Vector Store</span>
            <span class="status-val">ChromaDB</span>
        </div>
        <div class="status-row">
            <span class="status-key">Indexed PDFs</span>
            <span class="status-val">{num_files} Docs</span>
        </div>
        <div class="status-row">
            <span class="status-key">Embed Chunks</span>
            <span class="status-val">{chunk_count} Chunks</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Input Override if needed
    if not API_KEY or st.session_state.api_key_override:
        st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)
        user_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key_override,
            help="Retrieve a key from Google AI Studio: https://aistudio.google.com/",
            placeholder="Enter AIzaSy..."
        )
        if user_key != st.session_state.api_key_override:
            st.session_state.api_key_override = user_key
            st.rerun()
            
    # ---- Document Ingestion ----
    st.markdown('<div class="section-label">Document Ingestion</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Scheme PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload official government policy gazettes or scheme documentation."
    )
    
    process_btn = st.button("Process & Index Documents", type="primary", use_container_width=True)
    
    if process_btn:
        if not API_KEY:
            st.error("API Key not configured! Please enter your key above or configure secrets.")
        elif not uploaded_files:
            st.warning("Please select at least one PDF file prior to indexing.")
        else:
            with st.spinner("Parsing documents & generating vector embeddings..."):
                saved_paths = []
                for uploaded_file in uploaded_files:
                    if utils.validate_pdf(uploaded_file.name, uploaded_file.size):
                        path = utils.save_uploaded_file(uploaded_file, UPLOAD_DIR)
                        saved_paths.append(path)
                
                all_chunks = []
                for path in saved_paths:
                    try:
                        docs = pdf_processor.extract_text_from_pdf(path)
                        chunks = pdf_processor.chunk_documents(docs)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        st.error(f"Error reading {os.path.basename(path)}: {str(e)}")
                
                if all_chunks:
                    try:
                        db = vector_store.get_vector_store(CHROMA_DIR, API_KEY)
                        vector_store.add_documents_to_store(db, all_chunks)
                        st.session_state.vector_db = db
                        st.session_state.uploaded_files = [
                            f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.pdf')
                        ]
                        st.success(f"Successfully indexed {len(saved_paths)} PDFs into {len(all_chunks)} chunks.")
                    except Exception as e:
                        st.error(f"Database error: {str(e)}")
                else:
                    st.error("Could not extract readable text from uploaded PDF files.")
                    
    # Active Files Registry
    if num_files > 0:
        with st.expander("Indexed Document Archive"):
            for f in st.session_state.uploaded_files:
                st.caption(f"• {f}")
                
    st.markdown("---")
    
    # ---- Database Reset ----
    st.markdown('<div class="section-label">Archive Management</div>', unsafe_allow_html=True)
    reset_btn = st.button("Reset Database & Memory", type="secondary", use_container_width=True)
    if reset_btn:
        with st.spinner("Clearing local storage & memory..."):
            utils.clear_directory(UPLOAD_DIR)
            vector_store.reset_vector_store(CHROMA_DIR)
            st.session_state.chat_history = []
            st.session_state.vector_db = None
            st.session_state.uploaded_files = []
            st.session_state.pending_question = None
            st.session_state.query_cache = {}
            st.success("System reset successfully.")
            st.rerun()


# ===================== MAIN NEWSPAPER MASTHEAD =====================
st.markdown("""
<div class="masthead-container">
    <div class="masthead-meta-top">
        <span>PUBLIC POLICY INTELLIGENCE ARCHIVE</span>
        <span>EST. 2026 • NEW DELHI, INDIA</span>
        <span>LIVE GAZETTE EDITION</span>
    </div>
    <h1 class="masthead-title">JAN SAHAYAK</h1>
    <div class="masthead-subtitle">An Independent Digital Gazette & Grounded AI Intelligence System for Indian Government Policy</div>
</div>
""", unsafe_allow_html=True)


# ===================== TABS NAVIGATION =====================
tab_chat, tab_schemes, tab_howto = st.tabs([
    "INQUIRY DESK", 
    "SCHEME COMPENDIUM", 
    "SYSTEM METHODOLOGY"
])


# ===================== TAB 1: INQUIRY DESK (CHAT) =====================
with tab_chat:
    
    # Front-page editorial leadpiece if no chat history
    if not st.session_state.chat_history and st.session_state.pending_question is None:
        st.markdown("""
        <div class="editorial-lead-box">
            <div class="editorial-headline">Grounded Artificial Intelligence for Public Policy Verification</div>
            <div class="editorial-subhead">
                Jan Sahayak operates via strict Retrieval-Augmented Generation (RAG). 
                Every answer is compiled directly from official government scheme gazettes without extrapolation or external assumptions.
            </div>
            <div class="editorial-columns-3">
                <div class="editorial-col">
                    <div class="editorial-col-num">I. INDEXING</div>
                    <div class="editorial-col-title">Official Gazettes</div>
                    <div class="editorial-col-body">Government policy documents are uploaded, extracted page by page, and split into structured 1000-character segments.</div>
                </div>
                <div class="editorial-col">
                    <div class="editorial-col-num">II. RETRIEVAL</div>
                    <div class="editorial-col-title">Semantic Search</div>
                    <div class="editorial-col-body">User inquiries trigger a vector similarity search across ChromaDB to locate the top-3 most relevant source passages.</div>
                </div>
                <div class="editorial-col">
                    <div class="editorial-col-num">III. CITATION</div>
                    <div class="editorial-col-title">Grounded Synthesis</div>
                    <div class="editorial-col-body">Gemini 3.6 generates objective responses strictly bound to retrieved context, attaching complete source citations.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-label">Suggested Reference Inquiries</div>', unsafe_allow_html=True)
        
        preset_queries = [
            "What is PMJDY?",
            "What is the overdraft limit under PMJDY?",
            "Who is eligible for PM Kisan?",
            "What health coverage does Ayushman Bharat provide?"
        ]
        
        q_cols = st.columns(2)
        for idx, query_text in enumerate(preset_queries):
            with q_cols[idx % 2]:
                if st.button(f"QUERY #{idx+1}: {query_text}", key=f"preset_home_{idx}", use_container_width=True, type="secondary"):
                    st.session_state.pending_question = query_text
                    st.rerun()
                    
        st.markdown("<br>", unsafe_allow_html=True)

    # State verification notices
    is_ready = bool(API_KEY) and chunk_count > 0

    if not API_KEY:
        st.markdown("""
        <div class="editorial-notice">
            <b>NOTICE: API KEY REQUIRED</b> — Please enter your Google Gemini API Key in the left Editorial Control Panel to activate inquiry desk functionality.
        </div>
        """, unsafe_allow_html=True)
    elif num_files == 0:
        st.markdown("""
        <div class="editorial-notice">
            <b>NOTICE: ARCHIVE IS EMPTY</b> — No government scheme documents are currently indexed in the database.
            <br><br>
            <i>Instructions: Upload policy PDFs using the Editorial Control Panel on the left, or generate the default reference PDFs below.</i>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate & Index Official Reference Schemes", type="primary", use_container_width=True):
            with st.spinner("Generating official reference PDFs..."):
                try:
                    import generate_dummy_assets
                    generate_dummy_assets.ensure_directories([UPLOAD_DIR])
                    generate_dummy_assets.create_dummy_pdf(os.path.join(UPLOAD_DIR, "scheme_summary.pdf"))
                    generate_dummy_assets.create_ayushman_bharat_pdf(os.path.join(UPLOAD_DIR, "ayushman_bharat.pdf"))
                    generate_dummy_assets.create_dummy_video(os.path.join(UPLOAD_DIR, "demo_video.mp4"))
                    st.session_state.uploaded_files = [
                        f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.pdf')
                    ]
                    st.success("Reference PDFs generated. Click 'Process & Index Documents' in the sidebar to complete database build.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating assets: {str(e)}")
                    
    elif chunk_count == 0:
        st.markdown("""
        <div class="editorial-notice">
            <b>NOTICE: DOCUMENTS DETECTED BUT UNINDEXED</b> — PDF files exist in the uploads directory but have not been vectorized into ChromaDB.
            <br><br>
            Please click <b>"Process & Index Documents"</b> in the Editorial Control Panel.
        </div>
        """, unsafe_allow_html=True)

    # Active Chat Interface
    if is_ready:
        if st.session_state.vector_db is None:
            st.warning("Database initializing... please wait.")
            
        # Render Existing Chat Log
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "sources" in message and message["sources"]:
                    with st.expander("VERIFIED SOURCE FOOTNOTES & CITATIONS"):
                        for idx, src in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="footnote-box">
                                <div class="footnote-head">
                                    <span>DOCUMENT CITATION NO. {idx+1}: {src['source']} (PAGE {src['page']})</span>
                                    <span class="footnote-badge">RELEVANCE: {src['score']}%</span>
                                </div>
                                <div class="footnote-body">{src['content']}</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Input & Query Execution
        input_disabled = (st.session_state.vector_db is None or chunk_count == 0)
        placeholder_str = "Submit inquiry on any government scheme..." if not input_disabled else "Index documents to enable inquiry desk"
        
        user_query = st.chat_input(placeholder_str, disabled=input_disabled)
        
        # Check for pending question triggered from preset buttons
        if st.session_state.pending_question and not input_disabled:
            user_query = st.session_state.pending_question
            st.session_state.pending_question = None

        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            cache = st.session_state.query_cache
            if user_query in cache:
                answer = cache[user_query]["answer"]
                sources = cache[user_query]["sources"]
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Searching gazettes & compiling answer..."):
                        try:
                            chunks_with_scores = chatbot.retrieve_relevant_chunks(
                                st.session_state.vector_db,
                                user_query,
                                k=3
                            )

                            result = chatbot.generate_answer(
                                user_query,
                                chunks_with_scores,
                                API_KEY
                            )

                            answer = result["answer"]
                            sources = result["sources"]

                            if len(cache) >= 30:
                                cache.pop(next(iter(cache)))
                            cache[user_query] = {"answer": answer, "sources": sources}

                        except Exception as e:
                            answer = f"⚠️ **Execution Error:** {str(e)}\n\nPlease check your API key and network connection."
                            sources = []

            with st.chat_message("assistant"):
                st.markdown(answer)

                if sources:
                    with st.expander("VERIFIED SOURCE FOOTNOTES & CITATIONS"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"""
                            <div class="footnote-box">
                                <div class="footnote-head">
                                    <span>DOCUMENT CITATION NO. {idx+1}: {src['source']} (PAGE {src['page']})</span>
                                    <span class="footnote-badge">RELEVANCE: {src['score']}%</span>
                                </div>
                                <div class="footnote-body">{src['content']}</div>
                            </div>
                            """, unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            st.rerun()


# ===================== TAB 2: SCHEME COMPENDIUM =====================
with tab_schemes:
    st.markdown('<div class="section-label">Official Scheme Compendium & Reference Directory</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    scheme_cols = st.columns(3)
    for idx, scheme in enumerate(DEMO_SCHEMES):
        with scheme_cols[idx]:
            st.markdown(f"""
            <div class="scheme-editorial-card">
                <div>
                    <div class="scheme-editorial-code">{scheme['code']}</div>
                    <div class="scheme-editorial-title">{scheme['name']}</div>
                    <div class="scheme-editorial-body">{scheme['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Categorized Inquiry Index</div>', unsafe_allow_html=True)
    st.markdown("Click any inquiry below to submit it directly to the Central Inquiry Desk:")
    st.markdown("<br>", unsafe_allow_html=True)
    
    for scheme in DEMO_SCHEMES:
        st.markdown(f"""
        <div style="font-family: 'Newsreader', serif; font-size: 1.2rem; font-weight: 700; color: var(--ink-black); margin-bottom: 0.5rem; margin-top: 1rem;">
            {scheme['name']} <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 400; color: var(--text-subtle);">[{scheme['short']}]</span>
        </div>
        """, unsafe_allow_html=True)
        
        q_cols = st.columns(len(scheme["questions"]))
        for q_idx, q in enumerate(scheme["questions"]):
            with q_cols[q_idx]:
                if st.button(q, key=f"scheme_q_{scheme['short']}_{q_idx}", use_container_width=True, type="secondary"):
                    st.session_state.pending_question = q
                    st.rerun()


# ===================== TAB 3: SYSTEM METHODOLOGY =====================
with tab_howto:
    st.markdown('<div class="section-label">System Documentary & Instructional Guidance</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Documentary Video Dispatch
    st.markdown("""
    <div style="background: var(--surface-white); border: 1px solid var(--border-rule); border-top: 3px solid var(--ink-black); padding: 1.5rem; margin-bottom: 2rem;">
        <div style="font-family: 'Newsreader', serif; font-size: 1.4rem; font-weight: 700; color: var(--ink-black); margin-bottom: 0.5rem;">
            Video Dispatch: Public Policy & Scheme Intelligence Framework
        </div>
        <div style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 1.2rem; line-height: 1.6;">
            An orientation briefing on understanding public policy structures, document extraction protocols, and grounded AI inquiry.
        </div>
        <div style="position: relative; width: 100%; padding-bottom: 56.25%; height: 0; overflow: hidden; border: 1px solid var(--border-rule);">
            <iframe 
                src="https://www.youtube-nocookie.com/embed/h2aWGlSVr98" 
                title="Government Schemes Tutorial"
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-label">Operational Procedures</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    guide_cols = st.columns(2)
    for t_idx, tutorial in enumerate(TUTORIAL_VIDEOS):
        with guide_cols[t_idx]:
            steps_html = ''.join(
                f'<div style="font-size: 0.85rem; color: var(--ink-black); padding: 0.4rem 0.8rem; margin: 0.4rem 0; border-left: 2px solid var(--ink-black); background: var(--bg-paper);">▸ {step}</div>' 
                for step in tutorial['steps']
            )
            st.markdown(f"""
            <div style="background: var(--surface-white); border: 1px solid var(--border-rule); padding: 1.5rem; height: 100%;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 700; color: var(--ink-black); margin-bottom: 0.5rem;">{tutorial['title']}</div>
                <div style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 1rem; line-height: 1.6;">{tutorial['description']}</div>
                {steps_html}
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Technical Architecture & RAG Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: var(--surface-white); border: 1px solid var(--border-rule); border-top: 3px solid var(--ink-black); padding: 1.5rem; margin-top: 1rem;">
        <div style="font-family: 'Newsreader', serif; font-size: 1.3rem; font-weight: 700; color: var(--ink-black); margin-bottom: 1rem;">
            End-to-End Grounded Retrieval Pipeline
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; text-align: center;">
            <div style="border: 1px solid var(--border-rule); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--text-subtle);">STAGE 01</div>
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--ink-black); margin: 0.3rem 0;">PDF Ingestion</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Parsing official scheme gazettes</div>
            </div>
            <div style="border: 1px solid var(--border-rule); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--text-subtle);">STAGE 02</div>
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--ink-black); margin: 0.3rem 0;">Text Chunking</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">1000-char overlapping blocks</div>
            </div>
            <div style="border: 1px solid var(--border-rule); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--text-subtle);">STAGE 03</div>
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--ink-black); margin: 0.3rem 0;">ONNX Embedding</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Fast CPU MiniLM vectorization</div>
            </div>
            <div style="border: 1px solid var(--border-rule); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--text-subtle);">STAGE 04</div>
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--ink-black); margin: 0.3rem 0;">ChromaDB Index</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Persistent vector storage</div>
            </div>
            <div style="border: 1px solid var(--border-rule); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--text-subtle);">STAGE 05</div>
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--ink-black); margin: 0.3rem 0;">Cosine Search</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Top-K chunk retrieval</div>
            </div>
            <div style="border: 1px solid var(--border-rule); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--text-subtle);">STAGE 06</div>
                <div style="font-weight: 700; font-size: 0.9rem; color: var(--ink-black); margin: 0.3rem 0;">Gemini Synthesis</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Grounded answer + citations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
