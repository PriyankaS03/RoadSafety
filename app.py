"""
VigiRoad AI – Smart Vision-Based Road Safety & Safe Route System
Main entry point: Streamlit multi-page app
"""
import streamlit as st
from utils.db import init_db
from utils.auth import login_page

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VigiRoad AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (modern SaaS dashboard look) ───────────────────────────────────
st.markdown("""
<style>
/* ---- font & root ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { 
    background: rgba(255,255,255,0.05); 
    border-radius: 10px; padding: 8px 12px;
    margin: 3px 0; cursor: pointer;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.12); }

/* ---- main background ---- */
.main { background: #f8fafc; }

/* ---- metric cards ---- */
.metric-card {
    background: white; border-radius: 16px;
    padding: 20px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #e8edf2; margin-bottom: 12px;
}
.metric-card .value { font-size: 2.2rem; font-weight: 800; margin: 4px 0; }
.metric-card .label { font-size: 0.82rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
.metric-card .delta { font-size: 0.78rem; font-weight: 600; }

/* ---- alert cards ---- */
.alert-card {
    background: white; border-radius: 14px;
    padding: 16px 20px; margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #ef4444;
}
.alert-card.warning { border-left-color: #f59e0b; }
.alert-card.info    { border-left-color: #3b82f6; }
.alert-card.success { border-left-color: #10b981; }

/* ---- gradient buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white; border: none; border-radius: 10px;
    padding: 10px 24px; font-weight: 600; font-size: 0.9rem;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.45); }

/* ---- section headers ---- */
.section-header {
    font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;
}
.section-sub { color: #64748b; font-size: 0.92rem; margin-bottom: 24px; }

/* ---- route cards ---- */
.route-safe   { background: linear-gradient(135deg,#064e3b,#065f46); border-left: 5px solid #10b981; border-radius: 14px; padding: 16px; color: #e2e8f0; }
.route-medium { background: linear-gradient(135deg,#78350f,#92400e); border-left: 5px solid #f59e0b; border-radius: 14px; padding: 16px; color: #e2e8f0; }
.route-risky  { background: linear-gradient(135deg,#7f1d1d,#991b1b); border-left: 5px solid #ef4444; border-radius: 14px; padding: 16px; color: #e2e8f0; }

/* ---- badge ---- */
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
.badge-red    { background:#fee2e2; color:#dc2626; }
.badge-amber  { background:#fef3c7; color:#d97706; }
.badge-green  { background:#d1fae5; color:#059669; }
.badge-blue   { background:#dbeafe; color:#2563eb; }

/* hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DB init ───────────────────────────────────────────────────────────────────
init_db()

# ── Auth / routing ────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    login_page()
else:
    user = st.session_state.user
    role = user["role"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px'>
            <div style='font-size:2.4rem'>🛡️</div>
            <div style='font-size:1.1rem; font-weight:800; color:#f8fafc'>VigiRoad AI</div>
            <div style='font-size:0.72rem; color:#94a3b8; margin-top:2px'>Smart Road Safety System</div>
        </div>
        <hr style='border-color:#334155; margin:12px 0'>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:0.78rem; color:#94a3b8; padding:0 8px'>Logged in as</div>"
                    f"<div style='font-size:0.92rem; font-weight:700; color:#e2e8f0; padding:0 8px 12px'>"
                    f"👤 {user['name']} <span style='color:#818cf8'>({role})</span></div>", unsafe_allow_html=True)

        # Navigation per role
        nav_options = {
            "USER":      ["🗺️ Safe Route Finder", "📹 Live Detection", "🔬 Media Detection", "📊 Road Stats"],
            "POLICE":    ["🚨 Accident Alerts",   "📹 Live Detection", "🔬 Media Detection", "📊 Road Stats"],
            "AUTHORITY": ["🚧 Road Damage Reports","📹 Live Detection", "🔬 Media Detection", "📊 Road Stats"],
            "AMBULANCE": ["🚑 Emergency Alerts",   "📹 Live Detection", "🔬 Media Detection", "📊 Road Stats"],
        }
        choices = nav_options.get(role, nav_options["USER"])
        page = st.radio("Navigation", choices, label_visibility="collapsed")

        st.markdown("<hr style='border-color:#334155; margin:16px 0'>", unsafe_allow_html=True)

        # Demo simulation buttons
        st.markdown("<div style='font-size:0.8rem; color:#94a3b8; padding:0 8px 8px'>🎭 Demo Triggers</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💥 Accident", use_container_width=True):
                from utils.detection import simulate_event
                simulate_event("accident")
                st.toast("🚨 Accident simulated!", icon="💥")
        with col2:
            if st.button("🕳️ Pothole", use_container_width=True):
                from utils.detection import simulate_event
                simulate_event("pothole")
                st.toast("🕳️ Pothole simulated!", icon="🚧")

        st.markdown("<hr style='border-color:#334155; margin:16px 0'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    # ── Page routing ─────────────────────────────────────────────────────────
    if "Safe Route" in page:
        from pages.user_dashboard import show_user_dashboard
        show_user_dashboard()
    elif "Accident Alerts" in page:
        from pages.police_dashboard import show_police_dashboard
        show_police_dashboard()
    elif "Road Damage" in page:
        from pages.authority_dashboard import show_authority_dashboard
        show_authority_dashboard()
    elif "Emergency" in page:
        from pages.ambulance_dashboard import show_ambulance_dashboard
        show_ambulance_dashboard()
    elif "Live Detection" in page:
        from pages.live_detection import show_live_detection
        show_live_detection()
    elif "Media Detection" in page:
        from pages._media_detection import show_media_detection
        show_media_detection()
    elif "Road Stats" in page:
        from pages.stats_dashboard import show_stats_dashboard
        show_stats_dashboard()
