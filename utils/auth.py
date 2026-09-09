"""
utils/auth.py – Login & Registration page
Police & Ambulance users must specify their station location and controlling area.
"""
import streamlit as st
from utils.db import authenticate, register_user
from utils.emergency_services import POLICE_STATIONS, HOSPITALS

# Pre-built station options for quick selection
_POLICE_OPTIONS = {ps["name"]: ps for ps in POLICE_STATIONS}
_HOSPITAL_OPTIONS = {h["name"]: h for h in HOSPITALS}


def login_page():
    # Hide sidebar completely on login page
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { display: none !important; }
        button[data-testid="stSidebarCollapsedControl"],
        button[data-testid="baseButton-headerNoPadding"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Hero layout
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        min-height: 100vh; display: flex; align-items: center; justify-content: center;
        padding: 0; margin: -80px -80px 0 -80px;
    '>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style='text-align:center; padding: 40px 0 20px'>
            <div style='font-size:3.5rem'>🛡️</div>
            <h1 style='font-size:2rem; font-weight:900; color:#0f172a; margin:8px 0 2px'>VigiRoad AI</h1>
            <p style='color:#64748b; font-size:0.95rem'>Smart Vision-Based Road Safety & Safe Route System</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div style='background:white; border-radius:20px; padding:32px 36px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.12); border:1px solid #e2e8f0;'>
            """, unsafe_allow_html=True)

            tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])

            # ── Sign In Tab ──────────────────────────────────────────────
            with tab_login:
                email = st.text_input("📧 Email Address", placeholder="your@email.com", key="login_email")
                password = st.text_input("🔑 Password", type="password", placeholder="••••••••", key="login_pwd")

                if st.button("Sign In →", use_container_width=True, key="btn_login"):
                    if not email or not password:
                        st.error("Please enter email and password.")
                    else:
                        user = authenticate(email, password)
                        if user:
                            st.session_state.user = user
                            st.rerun()
                        else:
                            st.error("Invalid email or password. Please try again or create an account.")

            # ── Register Tab ─────────────────────────────────────────────
            with tab_register:
                reg_name  = st.text_input("👤 Full Name", placeholder="John Doe", key="reg_name")
                reg_email = st.text_input("📧 Email Address", placeholder="your@email.com", key="reg_email")
                reg_pwd   = st.text_input("🔑 Password", type="password", placeholder="Choose a strong password", key="reg_pwd")
                reg_pwd2  = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter password", key="reg_pwd2")
                reg_role  = st.selectbox("🎭 Role", ["USER", "POLICE", "AUTHORITY", "AMBULANCE"], key="reg_role")

                # ── Station / Location fields for POLICE & AMBULANCE ─────
                station_name = None
                station_lat = None
                station_lon = None
                coverage_km = 10.0

                if reg_role == "POLICE":
                    st.markdown("""
                    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                                padding:12px 14px;margin:8px 0;font-size:0.82rem;color:#1e40af'>
                        🚔 <b>Police Registration</b> — Select your station or enter your location.
                        You will only see incidents within your controlling area.
                    </div>
                    """, unsafe_allow_html=True)

                    station_mode = st.radio(
                        "Station Selection", ["Select from list", "Enter manually"],
                        horizontal=True, key="police_station_mode",
                        label_visibility="collapsed"
                    )

                    if station_mode == "Select from list":
                        selected_station = st.selectbox(
                            "🏢 Select Your Police Station",
                            list(_POLICE_OPTIONS.keys()),
                            key="police_station_select"
                        )
                        if selected_station:
                            ps = _POLICE_OPTIONS[selected_station]
                            station_name = ps["name"]
                            station_lat = ps["lat"]
                            station_lon = ps["lon"]
                            coverage_km = ps["coverage_km"]
                            st.info(f"📍 Location: {ps['lat']:.4f}, {ps['lon']:.4f} · Coverage: {ps['coverage_km']} km")
                    else:
                        station_name = st.text_input("🏢 Station Name", placeholder="e.g. Gandhipuram Police Station", key="police_stn_name")
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            station_lat = st.number_input("📍 Latitude", value=11.0168, format="%.4f", key="police_lat")
                        with pc2:
                            station_lon = st.number_input("📍 Longitude", value=76.9558, format="%.4f", key="police_lon")
                        coverage_km = st.slider("📡 Controlling Area (km radius)", 5, 25, 10, key="police_coverage")

                elif reg_role == "AMBULANCE":
                    st.markdown("""
                    <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                                padding:12px 14px;margin:8px 0;font-size:0.82rem;color:#166534'>
                        🚑 <b>Ambulance/Hospital Registration</b> — Select your hospital or enter your location.
                        You will only see incidents within your coverage area.
                    </div>
                    """, unsafe_allow_html=True)

                    hosp_mode = st.radio(
                        "Hospital Selection", ["Select from list", "Enter manually"],
                        horizontal=True, key="hosp_station_mode",
                        label_visibility="collapsed"
                    )

                    if hosp_mode == "Select from list":
                        selected_hosp = st.selectbox(
                            "🏥 Select Your Hospital",
                            list(_HOSPITAL_OPTIONS.keys()),
                            key="hosp_station_select"
                        )
                        if selected_hosp:
                            h = _HOSPITAL_OPTIONS[selected_hosp]
                            station_name = h["name"]
                            station_lat = h["lat"]
                            station_lon = h["lon"]
                            coverage_km = h["coverage_km"]
                            st.info(f"📍 Location: {h['lat']:.4f}, {h['lon']:.4f} · Coverage: {h['coverage_km']} km")
                    else:
                        station_name = st.text_input("🏥 Hospital / Base Name", placeholder="e.g. GKNM Hospital", key="hosp_stn_name")
                        hc1, hc2 = st.columns(2)
                        with hc1:
                            station_lat = st.number_input("📍 Latitude", value=11.0168, format="%.4f", key="hosp_lat")
                        with hc2:
                            station_lon = st.number_input("📍 Longitude", value=76.9558, format="%.4f", key="hosp_lon")
                        coverage_km = st.slider("📡 Coverage Area (km radius)", 5, 30, 10, key="hosp_coverage")

                if st.button("Create Account →", use_container_width=True, key="btn_register"):
                    if not reg_name or not reg_email or not reg_pwd:
                        st.error("Please fill in all fields.")
                    elif reg_pwd != reg_pwd2:
                        st.error("Passwords do not match.")
                    elif len(reg_pwd) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif reg_role in ("POLICE", "AMBULANCE") and not station_lat:
                        st.error("Please select or enter your station/hospital location.")
                    else:
                        ok, msg = register_user(
                            reg_name, reg_email, reg_pwd, reg_role,
                            station_name=station_name,
                            station_lat=station_lat,
                            station_lon=station_lon,
                            coverage_km=coverage_km,
                        )
                        if ok:
                            st.success(msg + " You can now sign in.")
                        else:
                            st.error(msg)

            st.markdown("</div>", unsafe_allow_html=True)

        # Feature cards
        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        for col, icon, title, desc in [
            (f1, "🎯", "AI Detection", "Real-time pothole & accident detection via camera"),
            (f2, "🗺️", "Safe Routes", "AI-scored routes avoiding hazard zones"),
            (f3, "🚨", "Live Alerts", "Instant notifications to Police & Ambulance"),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:white; border-radius:14px; padding:18px;
                            box-shadow:0 4px 16px rgba(0,0,0,0.07); text-align:center;
                            border:1px solid #e8edf2;'>
                    <div style='font-size:1.8rem'>{icon}</div>
                    <div style='font-weight:700; font-size:0.9rem; color:#0f172a; margin:6px 0 3px'>{title}</div>
                    <div style='font-size:0.75rem; color:#64748b'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
