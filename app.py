import os
import base64
import textwrap

# Fix protobuf compiler descriptor compatibility issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Suppress transformers/tokenizers verbose startup warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

try:
    import streamlit as st
except ImportError as e:
    raise ImportError("Streamlit is required. Please run 'pip install streamlit'.") from e

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

import utils
import pdf_processor
import vector_store
import chatbot

# Page configuration — MUST be the first Streamlit command
st.set_page_config(
    page_title="JAN SAHAYAK — Public Policy Gazette & Research Desk",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env
load_dotenv()

# Load logo as base64 for embedding in HTML
def _load_logo_b64() -> str:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.jpg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

LOGO_B64 = _load_logo_b64()

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
            "Select scheme PDFs via the Document Utility Desk",
            "Execute 'Process & Index Documents' command",
            "Verify chunk creation and database count status",
            "Initiate grounded semantic search queries"
        ]
    },
    {
        "title": "Query Formulation & Citation Audit",
        "description": "Guidelines for formulating natural language inquiries and auditing returned source citations for fact verification.",
        "steps": [
            "Submit inquiry via the Central Research Desk",
            "Review grounded answer compiled by Gemini 3.6",
            "Inspect 'Source Evidence & Document References'",
            "Cross-check page numbers and confidence scores"
        ]
    }
]

# ===================== REFINED EDITORIAL NEWSPAPER STYLING =====================
st.markdown(textwrap.dedent("""
<style>
    /* ===== Typography Import ===== */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ===== Color Palette ===== */
    :root {
        --bg-paper: #F5F3EE;
        --surface-white: #FAF9F6;
        --text-main: #111111;
        --text-muted: #68645D;
        --text-subtle: #8C877D;
        --border-thin: #C9C4BA;
        --border-heavy: #151515;
    }

    /* ===== Global Resets ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-main) !important;
        background-color: var(--bg-paper) !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: var(--bg-paper) !important;
    }

    /* Editorial Max Width */
    .block-container {
        max-width: 1380px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* ===== Masthead ===== */
    .masthead-frame {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .masthead-logo {
        display: block;
        margin: 0 auto 0.6rem auto;
        width: 110px;
        height: 110px;
        object-fit: contain;
        filter: grayscale(100%) contrast(1.15);
        opacity: 0.90;
        mix-blend-mode: multiply;
        border: none;
        background: transparent;
    }
    .masthead-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border-thin);
        padding-bottom: 0.35rem;
        margin-bottom: 1rem;
    }
    .masthead-brand {
        font-family: 'Cormorant Garamond', 'Libre Baskerville', Georgia, serif;
        font-size: 3.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-main);
        margin: 0;
        line-height: 1.0;
        text-transform: uppercase;
    }
    .masthead-descriptor {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 1.15rem;
        font-style: italic;
        color: var(--text-muted);
        margin-top: 0.35rem;
        margin-bottom: 0.8rem;
    }
    .masthead-rules {
        border-top: 1px solid var(--border-thin);
        border-bottom: 2px solid var(--border-heavy);
        padding: 0.2rem 0;
        margin-bottom: 1rem;
    }

    /* ===== Sidebar (Publication Utility Rail) ===== */
    section[data-testid="stSidebar"] {
        background-color: #EDEAE4 !important;
        border-right: 1px solid var(--border-thin) !important;
        max-width: 290px !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }
    /* Restore correct button text colors inside sidebar (overrides the wildcard above) */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] * {
        color: var(--surface-white) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] p,
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] * {
        color: var(--text-main) !important;
    }
    .sidebar-head {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.35rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.1rem;
    }
    .sidebar-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted) !important;
        margin-bottom: 0.8rem;
    }

    .section-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--text-main) !important;
        border-bottom: 1px solid var(--border-thin);
        padding-bottom: 0.25rem;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    /* System Status Metadata Rows */
    .status-table {
        margin-bottom: 0.8rem;
        font-size: 0.78rem;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 0.28rem 0;
        border-bottom: 1px solid #E0DCD4;
    }
    .status-row:last-child {
        border-bottom: none;
    }
    .status-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-main);
    }

    /* ===== Restrained Buttons ===== */
    .stButton > button {
        border-radius: 2px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.12s ease-in-out !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="primary"] {
        background-color: var(--text-main) !important;
        color: var(--surface-white) !important;
        border: 1px solid var(--text-main) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2D2C2A !important;
        border-color: #2D2C2A !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="secondary"] {
        background-color: var(--surface-white) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-thin) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--bg-paper) !important;
        border-color: var(--text-main) !important;
        color: var(--text-main) !important;
    }

    /* ===== Form Inputs ===== */
    .stTextInput input {
        border-radius: 2px !important;
        border: 1px solid var(--border-thin) !important;
        background-color: var(--surface-white) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput input:focus {
        border-color: var(--text-main) !important;
        box-shadow: none !important;
    }

    /* ===== Streamlit Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        border-bottom: 2px solid var(--border-heavy) !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 0.5rem 1.6rem !important;
        color: var(--text-muted) !important;
        background-color: transparent !important;
        border-radius: 0px !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-main) !important;
        border-bottom: 3px solid var(--text-main) !important;
        background-color: transparent !important;
    }

    /* ===== Un-boxed Hero Section ===== */
    .hero-container {
        text-align: center;
        padding: 1rem 0 2.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-thin);
    }
    .hero-title {
        display: block;
        font-family: 'Cormorant Garamond', 'Libre Baskerville', Georgia, serif;
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1.15;
        color: var(--text-main);
        max-width: 920px;
        margin: 0 auto 0.8rem auto;
        text-align: center;
    }
    .hero-deck {
        display: block;
        font-family: 'Inter', sans-serif;
        font-size: 0.98rem;
        line-height: 1.6;
        color: var(--text-muted);
        max-width: 780px;
        margin: 0 auto 1.5rem auto;
        text-align: center;
    }

    /* ===== Editorial 3-Column Methodology ===== */
    .methodology-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0;
        margin-top: 1.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border-thin);
    }
    .methodology-col {
        padding: 0 1.8rem;
        border-right: 1px solid var(--border-thin);
        text-align: left;
    }
    .methodology-col:first-child { padding-left: 0; }
    .methodology-col:last-child { border-right: none; padding-right: 0; }
    
    .methodology-num {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
        color: var(--text-main);
        margin-bottom: 0.2rem;
    }
    .methodology-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--text-subtle);
        margin-bottom: 0.3rem;
    }
    .methodology-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.4rem;
    }
    .methodology-desc {
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--text-muted);
    }

    /* ===== Scheme Compendium Cards ===== */
    .scheme-card-flat {
        background: var(--surface-white);
        border: 1px solid var(--border-thin);
        padding: 1.4rem;
        height: 100%;
    }
    .scheme-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-subtle);
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .scheme-title {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .scheme-desc {
        font-size: 0.86rem;
        line-height: 1.6;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }

    /* ===== Research Brief & Inquiry Answer Styling ===== */
    .research-brief {
        background: var(--surface-white);
        border: 1px solid var(--border-thin);
        border-top: 3px solid var(--border-heavy);
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .brief-head {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: var(--text-subtle);
        text-transform: uppercase;
        border-bottom: 1px solid var(--border-thin);
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }

    /* Source Citation Footnotes */
    .citation-box {
        background: var(--bg-paper);
        border: 1px solid var(--border-thin);
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .citation-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-main);
        border-bottom: 1px solid var(--border-thin);
        padding-bottom: 0.3rem;
        margin-bottom: 0.4rem;
    }
    .citation-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        line-height: 1.55;
        color: var(--text-muted);
        white-space: pre-wrap;
    }

    /* Custom Chat Input */
    div[data-testid="stChatInput"] {
        border-radius: 2px !important;
        border: 1px solid var(--border-heavy) !important;
        background: var(--surface-white) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-main) !important;
    }
    .stBottom, div[data-testid="stBottom"] {
        background-color: var(--bg-paper) !important;
        border-top: 1px solid var(--border-thin) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-paper) !important;
        border: 1px solid var(--border-thin) !important;
        border-radius: 2px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: var(--text-main) !important;
    }

    hr {
        border-color: var(--border-thin) !important;
        margin: 1.4rem 0 !important;
    }

    .notice-box {
        background: var(--surface-white);
        border: 1px solid var(--border-thin);
        border-left: 3px solid var(--border-heavy);
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--text-main);
    }
</style>
"""), unsafe_allow_html=True)

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


# ===================== SIDEBAR (Publication Utility Rail) =====================
with st.sidebar:
    st.markdown(textwrap.dedent("""
    <div style="padding-top: 0.4rem;">
        <div class="sidebar-head">Utility Desk</div>
        <div class="sidebar-sub">System Registry & Ingestion</div>
    </div>
    """), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---- System Status Metadata Rows ----
    is_connected = bool(API_KEY)
    status_str = "CONNECTED" if is_connected else "DISCONNECTED"
    
    chunk_count = 0
    if st.session_state.vector_db is not None:
        chunk_count = vector_store.get_chunk_count(st.session_state.vector_db)
    num_files = len(st.session_state.uploaded_files)
    
    st.markdown(textwrap.dedent(f"""
    <div class="section-label">System Status</div>
    <div class="status-table">
        <div class="status-row">
            <span class="status-label">AI Service</span>
            <span class="status-value">{status_str}</span>
        </div>
        <div class="status-row">
            <span class="status-label">Model</span>
            <span class="status-value">Gemini 3.6 Flash</span>
        </div>
        <div class="status-row">
            <span class="status-label">Vector Store</span>
            <span class="status-value">ChromaDB</span>
        </div>
        <div class="status-row">
            <span class="status-label">Indexed</span>
            <span class="status-value">{num_files} DOCS</span>
        </div>
        <div class="status-row">
            <span class="status-label">Chunks</span>
            <span class="status-value">{chunk_count}</span>
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    # API Key Input Override if needed
    if not API_KEY or st.session_state.api_key_override:
        st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)
        user_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key_override,
            help="Retrieve a key from Google AI Studio: https://aistudio.google.com/",
            placeholder="AIzaSy..."
        )
        if user_key != st.session_state.api_key_override:
            st.session_state.api_key_override = user_key
            st.rerun()
            
    # ---- Document Desk ----
    st.markdown('<div class="section-label">Document Desk</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Scheme PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload official government policy guidelines."
    )
    
    process_btn = st.button("Process & Index Documents", type="primary", use_container_width=True)
    
    if process_btn:
        if not API_KEY:
            st.error("API Key not configured! Please enter your key above or in secrets.")
        elif not uploaded_files:
            st.warning("Please select at least one PDF file prior to indexing.")
        else:
            with st.spinner("Extracting text & vectorizing chunks..."):
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
                        st.success(f"Indexed {len(saved_paths)} PDFs into {len(all_chunks)} chunks.")
                    except Exception as e:
                        st.error(f"Database error: {str(e)}")
                else:
                    st.error("Could not extract readable text from uploaded PDF files.")
                    
    # Active Files Registry
    if num_files > 0:
        with st.expander("Indexed Files Archive"):
            for f in st.session_state.uploaded_files:
                st.caption(f"• {f}")
                
    st.markdown("---")
    
    # ---- Database Reset ----
    st.markdown('<div class="section-label">Archive Reset</div>', unsafe_allow_html=True)
    reset_btn = st.button("Reset Memory & DB", type="secondary", use_container_width=True)
    if reset_btn:
        with st.spinner("Resetting storage & cache..."):
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
_logo_html = f'<img src="data:image/jpeg;base64,{LOGO_B64}" class="masthead-logo" alt="Jan Sahayak emblem">' if LOGO_B64 else ''
st.markdown(
'<div class="masthead-frame">'
'<div class="masthead-meta">'
'<span>PUBLIC POLICY RESEARCH DESK</span>'
'<span>EST. 2026 • NEW DELHI, INDIA</span>'
'<span>ISSUE 04 • LIVE GAZETTE</span>'
'</div>'
+ _logo_html +
'<div class="masthead-brand">JAN SAHAYAK</div>'
'<div class="masthead-descriptor">An independent digital gazette and grounded AI intelligence system for Indian government policy.</div>'
'<div class="masthead-rules"></div>'
'</div>',
unsafe_allow_html=True)


# ===================== TABS NAVIGATION =====================
tab_chat, tab_schemes, tab_howto = st.tabs([
    "INQUIRY DESK", 
    "SCHEME COMPENDIUM", 
    "SYSTEM METHODOLOGY"
])


# ===================== TAB 1: INQUIRY DESK (RESEARCH DESK) =====================
with tab_chat:
    
    # Un-boxed Hero Section if no chat history
    if not st.session_state.chat_history and st.session_state.pending_question is None:
        st.markdown(
'<div class="hero-container">'
'<div class="hero-title">Grounded Artificial Intelligence for Public Policy Verification</div>'
'<div class="hero-deck">Jan Sahayak operates via strict Retrieval-Augmented Generation (RAG). '
'Every answer is compiled directly from official government scheme gazettes without extrapolation or external assumptions.</div>'
'<div class="methodology-grid">'
'<div class="methodology-col">'
'<div class="methodology-num">01</div>'
'<div class="methodology-label">OFFICIAL GAZETTES</div>'
'<div class="methodology-title">INDEXING</div>'
'<div class="methodology-desc">Government policy documents are uploaded, extracted page by page, and split into structured 1000-character segments.</div>'
'</div>'
'<div class="methodology-col">'
'<div class="methodology-num">02</div>'
'<div class="methodology-label">SEMANTIC SEARCH</div>'
'<div class="methodology-title">RETRIEVAL</div>'
'<div class="methodology-desc">User inquiries trigger a vector similarity search across ChromaDB to locate the top-3 most relevant source passages.</div>'
'</div>'
'<div class="methodology-col">'
'<div class="methodology-num">03</div>'
'<div class="methodology-label">GROUNDED SYNTHESIS</div>'
'<div class="methodology-title">CITATION</div>'
'<div class="methodology-desc">Gemini 3.6 generates objective responses strictly bound to retrieved context, attaching complete source citations.</div>'
'</div>'
'</div>'
'</div>',
unsafe_allow_html=True)
        
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
        st.markdown('<div class="notice-box"><b>API KEY REQUIRED:</b> Please enter your Google Gemini API Key in the left Utility Desk to activate the inquiry desk.</div>', unsafe_allow_html=True)
    elif num_files == 0:
        st.markdown('<div class="notice-box"><b>ARCHIVE IS EMPTY:</b> No government scheme documents are currently indexed in the database.<br><br><i>Upload policy PDFs using the Document Desk on the left, or generate the default reference schemes below.</i></div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="notice-box"><b>DOCUMENTS UNINDEXED:</b> PDF files exist in the uploads directory but have not been vectorized into ChromaDB.<br><br>Please click <b>"Process &amp; Index Documents"</b> in the left Utility Desk.</div>', unsafe_allow_html=True)

    # Active Inquiry Research Brief History
    if is_ready:
        if st.session_state.vector_db is None:
            st.warning("Database initializing... please wait.")
            
        # Render Research Brief History
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem; letter-spacing: 1.5px; color: var(--text-subtle); text-transform: uppercase; margin-top: 1.2rem; margin-bottom: 0.2rem;">SUBMITTED INQUIRY</div><div style="font-family: \'Cormorant Garamond\', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); border-bottom: 1px solid var(--border-thin); padding-bottom: 0.5rem; margin-bottom: 1rem;">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="research-brief"><div class="brief-head">POLICY BRIEF &amp; GROUNDED SYNTHESIS</div><div style="font-size: 0.92rem; line-height: 1.7; color: var(--text-main);">{message["content"]}</div></div>', unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    with st.expander("SOURCE EVIDENCE & DOCUMENT REFERENCES"):
                        for idx, src in enumerate(message["sources"]):
                            st.markdown(f'<div class="citation-box"><div class="citation-head"><span>[REF {idx+1}] {src["source"]} — PAGE {src["page"]}</span><span>RELEVANCE SCORE: {src["score"]}%</span></div><div class="citation-text">{src["content"]}</div></div>', unsafe_allow_html=True)

        # Input & Query Execution
        input_disabled = (st.session_state.vector_db is None or chunk_count == 0)
        placeholder_str = "Submit inquiry on any government scheme..." if not input_disabled else "Index documents to enable research desk"
        
        user_query = st.chat_input(placeholder_str, disabled=input_disabled)
        
        # Check for pending question triggered from preset buttons
        if st.session_state.pending_question and not input_disabled:
            user_query = st.session_state.pending_question
            st.session_state.pending_question = None

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            cache = st.session_state.query_cache
            if user_query in cache:
                answer = cache[user_query]["answer"]
                sources = cache[user_query]["sources"]
            else:
                with st.spinner("Searching gazettes & compiling policy brief..."):
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
                        answer = f"⚠️ Execution Error: {str(e)}"
                        sources = []

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
            st.markdown(textwrap.dedent(f"""
            <div class="scheme-card-flat">
                <div class="scheme-code">{scheme['code']}</div>
                <div class="scheme-title">{scheme['name']}</div>
                <div class="scheme-desc">{scheme['description']}</div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Categorized Inquiry Index</div>', unsafe_allow_html=True)
    st.markdown("Select any inquiry below to submit it directly to the Central Research Desk:")
    st.markdown("<br>", unsafe_allow_html=True)
    
    for scheme in DEMO_SCHEMES:
        st.markdown(textwrap.dedent(f"""
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem; margin-top: 1rem;">
            {scheme['name']} <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 400; color: var(--text-subtle);">[{scheme['short']}]</span>
        </div>
        """), unsafe_allow_html=True)
        
        q_cols = st.columns(len(scheme["questions"]))
        for q_idx, q in enumerate(scheme["questions"]):
            with q_cols[q_idx]:
                if st.button(q, key=f"scheme_q_{scheme['short']}_{q_idx}", use_container_width=True, type="secondary"):
                    st.session_state.pending_question = q
                    st.rerun()


# ===================== TAB 3: SYSTEM METHODOLOGY =====================
with tab_howto:
    st.markdown('<div class="section-label">System Documentary & Instructional Briefing</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Documentary Video Frame
    st.markdown(textwrap.dedent("""
    <div style="background: var(--surface-white); border: 1px solid var(--border-thin); padding: 1.5rem; margin-bottom: 2rem;">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.4rem;">
            Briefing Video: Public Policy Document Indexing & Grounded Verification
        </div>
        <div style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 1.2rem; line-height: 1.6;">
            Orientation dispatch on policy extraction protocols, vector store searching, and document citation auditing.
        </div>
        <div style="position: relative; width: 100%; padding-bottom: 56.25%; height: 0; overflow: hidden; border: 1px solid var(--border-thin);">
            <iframe 
                src="https://www.youtube-nocookie.com/embed/h2aWGlSVr98" 
                title="Government Schemes Tutorial"
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-label">Operational Procedures</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    guide_cols = st.columns(2)
    for t_idx, tutorial in enumerate(TUTORIAL_VIDEOS):
        with guide_cols[t_idx]:
            steps_html = ''.join(
                f'<div style="font-size: 0.84rem; color: var(--text-main); padding: 0.4rem 0.8rem; margin: 0.35rem 0; border-left: 2px solid var(--border-heavy); background: var(--bg-paper);">▸ {step}</div>' 
                for step in tutorial['steps']
            )
            st.markdown(textwrap.dedent(f"""
            <div style="background: var(--surface-white); border: 1px solid var(--border-thin); padding: 1.5rem; height: 100%;">
                <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.4rem;">{tutorial['title']}</div>
                <div style="font-size: 0.86rem; color: var(--text-muted); margin-bottom: 1rem; line-height: 1.6;">{tutorial['description']}</div>
                {steps_html}
            </div>
            """), unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Technical Architecture & Grounded Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown(textwrap.dedent("""
    <div style="background: var(--surface-white); border: 1px solid var(--border-thin); padding: 1.5rem; margin-top: 1rem;">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; font-weight: 700; color: var(--text-main); margin-bottom: 1rem;">
            End-to-End Grounded Retrieval Pipeline
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; text-align: center;">
            <div style="border: 1px solid var(--border-thin); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; color: var(--text-subtle);">STAGE 01</div>
                <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-main); margin: 0.3rem 0;">PDF Ingestion</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Parsing official scheme gazettes</div>
            </div>
            <div style="border: 1px solid var(--border-thin); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; color: var(--text-subtle);">STAGE 02</div>
                <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-main); margin: 0.3rem 0;">Text Chunking</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">1000-char overlapping blocks</div>
            </div>
            <div style="border: 1px solid var(--border-thin); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; color: var(--text-subtle);">STAGE 03</div>
                <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-main); margin: 0.3rem 0;">ONNX Embedding</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Fast CPU MiniLM vectorization</div>
            </div>
            <div style="border: 1px solid var(--border-thin); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; color: var(--text-subtle);">STAGE 04</div>
                <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-main); margin: 0.3rem 0;">ChromaDB Index</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Persistent vector storage</div>
            </div>
            <div style="border: 1px solid var(--border-thin); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; color: var(--text-subtle);">STAGE 05</div>
                <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-main); margin: 0.3rem 0;">Cosine Search</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Top-K chunk retrieval</div>
            </div>
            <div style="border: 1px solid var(--border-thin); padding: 1rem; background: var(--bg-paper);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; color: var(--text-subtle);">STAGE 06</div>
                <div style="font-weight: 700; font-size: 0.88rem; color: var(--text-main); margin: 0.3rem 0;">Gemini Synthesis</div>
                <div style="font-size: 0.76rem; color: var(--text-muted);">Grounded answer + citations</div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)
