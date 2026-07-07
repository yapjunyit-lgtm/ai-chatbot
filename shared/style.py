"""
Dark theme CSS matching ai-dashboard design for all Streamlit apps.
Import: from shared.style import inject_css, header, footer
"""
import streamlit as st


def inject_css():
    """Inject dark theme CSS matching the AI Studio dashboard design."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Base Dark Theme ── */
        .stApp { background: #0a0a14 !important; }
        #MainMenu, footer:not(.app-footer), .stDeployButton, header[data-testid="stHeader"] {
            visibility: hidden !important; display: none !important;
        }
        html, body, .stApp, [class*="css"], .stMarkdown, .stText, p, span, div, label, button {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            color: #f1f5f9;
        }
        code, pre, .stCode { font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }

        /* ── Main Content ── */
        .main .block-container { max-width: 900px !important; padding: 1rem 1.5rem 0 !important; }
        [data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: #12121f !important; border-right: 1px solid rgba(255,255,255,0.06) !important;
        }
        section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
        section[data-testid="stSidebar"] .stTextInput input {
            background: #1a1a2e !important; border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 10px !important; color: #f1f5f9 !important;
            font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
        }
        section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.06) !important; }

        /* ── Cards / Containers ── */
        .stChatMessage, div[data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 16px !important; padding: 1rem !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 12px !important; padding: 1rem !important;
        }
        div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
        div[data-testid="stMetric"] p { color: #f1f5f9 !important; font-weight: 700 !important; }
        .stExpander details {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 16px !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
            border-radius: 10px !important; transition: all 150ms ease !important;
        }
        .stButton > button[kind="primary"] { background: #6366f1 !important; border: none !important; }
        .stButton > button[kind="primary"]:hover { background: #4f46e5 !important; }
        .stButton > button[kind="secondary"] {
            background: transparent !important; border: 1px solid rgba(255,255,255,0.08) !important;
        }
        .stButton > button[kind="secondary"]:hover { background: rgba(255,255,255,0.05) !important; }

        /* ── Inputs ── */
        .stTextInput input, .stTextArea textarea, .stSelectbox > div {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 10px !important; color: #f1f5f9 !important;
            transition: border-color 150ms ease !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        }
        .stTextArea textarea { font-size: 0.9rem !important; }

        /* ── File Uploader ── */
        .stFileUploader section {
            border: 2px dashed rgba(255,255,255,0.08) !important;
            border-radius: 16px !important; padding: 1.5rem !important;
            background: rgba(255,255,255,0.02) !important;
        }
        .stFileUploader section:hover {
            border-color: #6366f1 !important; background: rgba(99,102,241,0.05) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #6366f1 !important; border-bottom-color: #6366f1 !important;
        }

        /* ── Spinner ── */
        .stSpinner > div { border-top-color: #6366f1 !important; }

        /* ── Select / Slider ── */
        .stSelectbox [data-baseweb="select"] { background: #1a1a2e !important; }
        .stSlider [data-baseweb="slider"] { accent-color: #6366f1; }

        /* ── Dataframe ── */
        .stDataFrame, [data-testid="stTable"] {
            border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.06) !important;
        }

        /* ── Warnings / Errors / Success ── */
        .stAlert { border-radius: 10px !important; }
        div[data-testid="stNotification"] { border-radius: 10px !important; }

        /* ── Code blocks ── */
        .stCodeBlock { background: #1e293b !important; border-radius: 10px !important; }
        .stCodeBlock code { color: #e2e8f0 !important; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

        /* ── Header ── */
        .app-header { text-align: center; padding: 1.5rem 0 1rem; }
        .app-header .icon { font-size: 2.5rem; display: block; margin-bottom: 0.5rem; }
        .app-header h1 { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; margin: 0; }
        .app-header p { color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem; }

        /* ── Footer ── */
        .app-footer { text-align: center; padding: 1.5rem 0; color: #64748b; font-size: 0.8rem;
            border-top: 1px solid rgba(255,255,255,0.06); margin-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)


def header(icon, title, subtitle):
    """Render app header matching the dashboard design."""
    st.markdown(f"""
    <div class="app-header">
        <span class="icon">{icon}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def footer(text=""):
    """Render app footer."""
    st.markdown(f'<div class="app-footer">{text}</div>', unsafe_allow_html=True)
