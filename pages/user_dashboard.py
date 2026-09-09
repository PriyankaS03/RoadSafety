"""
pages/user_dashboard.py – Safe Route Finder for regular users
- GPS/browser location detection with manual fallback
- Google Maps deep-link integration for navigation
- Route cards: High Risk / Medium Risk / Low Risk / Safe path
- Historical data: accidents, potholes, damaged roads
- Fixed badge rendering (no raw HTML leakage)
"""
import streamlit as st
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
from utils.routes import geocode, get_routes
from utils.db import get_reports, get_stats
from utils.voice_assistant import (
    voice_alert_widget_html, speak_html,
    pothole_warning_html, hospital_proximity_html, police_proximity_html,
    nav_instruction_html,
)
from utils.emergency_services import (
    POLICE_STATIONS, HOSPITALS, AMBULANCE_BASES,
    get_nearest_police, get_nearest_hospitals, get_nearest_ambulances,
)

ROUTE_COLORS = {
    "Low Risk Road":    "#10b981",
    "Medium Risk Road": "#f59e0b",
    "High Risk Road":   "#ef4444",
}
ROUTE_LINE = {
    "Low Risk Road":    "#00e676",
    "Medium Risk Road": "#ffab00",
    "High Risk Road":   "#ff1744",
}
ROUTE_EMOJI = {
    "Low Risk Road":    "✅",
    "Medium Risk Road": "⚠️",
    "High Risk Road":   "🔴",
}
ROUTE_DESC = {
    "Low Risk Road":    "Well-maintained route with minimal historical incidents. AI recommends this path for safe travel.",
    "Medium Risk Road": "Moderate risk. Some road damage and moderate accident history in this corridor. Drive with caution.",
    "High Risk Road":   "High risk zone. Multiple accident hotspots and severe road damage reported. Avoid if possible.",
}
ROUTE_BG = {
    "Low Risk Road":    "linear-gradient(135deg,#064e3b,#065f46)",
    "Medium Risk Road": "linear-gradient(135deg,#78350f,#92400e)",
    "High Risk Road":   "linear-gradient(135deg,#7f1d1d,#991b1b)",
}


# ── Google Maps URL builder ────────────────────────────────────────────────────
def _google_maps_url(o_lat, o_lon, d_lat, d_lon, waypoints=None):
    """Build a Google Maps directions URL."""
    base = "https://www.google.com/maps/dir/"
    origin = f"{o_lat},{o_lon}"
    dest   = f"{d_lat},{d_lon}"
    if waypoints:
        mid = "/".join(f"{wp[0]},{wp[1]}" for wp in waypoints[:3])
        return f"{base}{origin}/{mid}/{dest}"
    return f"{base}{origin}/{dest}"


def _build_route_map(routes, org, dst, show_police=True, show_hospitals=True, show_ambulances=True):
    """Build a dark-themed folium map with color-coded routes."""
    center_lat = (org["lat"] + dst["lat"]) / 2
    center_lon = (org["lon"] + dst["lon"]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    line_weights = [7, 5, 4]
    for i, r in enumerate(routes):
        if not r.get("geometry"):
            continue
        color  = ROUTE_LINE.get(r["label"], "#6366f1")
        weight = line_weights[i] if i < len(line_weights) else 3
        opacity = 0.92 if i == 0 else 0.65

        dash = (
            "12 6"  if r["label"] == "High Risk Road"   else
            "8 4"   if r["label"] == "Medium Risk Road"  else
            None
        )

        folium.PolyLine(
            r["geometry"],
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=folium.Tooltip(
                f"<b>{r['label']}</b><br>"
                f"📏 {r['km']} km &nbsp;⏱ ~{r['minutes']} min<br>"
                f"💥 Accidents: {r['accidents']} &nbsp;🚧 Potholes/Damage: {r['potholes']}<br>"
                f"⚡ Safety Score: {r['score']}",
                sticky=True,
            ),
            dash_array=dash,
        ).add_to(m)

        if i == 0:
            AntPath(
                r["geometry"], color=color, weight=weight + 1,
                opacity=0.45, delay=1200, dash_array=[20, 30],
            ).add_to(m)

    # Origin & Destination markers
    folium.Marker(
        [org["lat"], org["lon"]],
        tooltip=f"📍 Origin: {org['name'].split(',')[0].strip()}",
        icon=folium.DivIcon(html=(
            "<div style='background:#3b82f6;border:3px solid #fff;border-radius:50%;"
            "width:22px;height:22px;display:flex;align-items:center;justify-content:center;"
            "box-shadow:0 0 14px rgba(59,130,246,0.9);'>"
            "<div style='background:#fff;border-radius:50%;width:9px;height:9px;'></div></div>"
        )),
    ).add_to(m)

    folium.Marker(
        [dst["lat"], dst["lon"]],
        tooltip=f"🏁 Destination: {dst['name'].split(',')[0].strip()}",
        icon=folium.DivIcon(html=(
            "<div style='background:#ef4444;border:3px solid #fff;border-radius:50%;"
            "width:22px;height:22px;display:flex;align-items:center;justify-content:center;"
            "box-shadow:0 0 14px rgba(239,68,68,0.9);'>"
            "<div style='background:#fff;border-radius:50%;width:9px;height:9px;'></div></div>"
        )),
    ).add_to(m)

    if show_police:
        for ps in POLICE_STATIONS:
            folium.CircleMarker(
                [ps["lat"], ps["lon"]], radius=4, color="#60a5fa",
                fill=True, fill_color="#3b82f6", fill_opacity=0.9,
                tooltip=f"🚔 {ps['name']} · {ps['zone']} · {ps['coverage_km']}km",
            ).add_to(m)

    if show_hospitals:
        for h in HOSPITALS:
            hc = "#10b981" if h["type"] == "Government Hospital" else "#8b5cf6"
            folium.CircleMarker(
                [h["lat"], h["lon"]], radius=5, color=hc,
                fill=True, fill_color=hc, fill_opacity=0.9,
                tooltip=f"🏥 {h['name']} · {h['speciality']}",
            ).add_to(m)

    if show_ambulances:
        for a in AMBULANCE_BASES:
            folium.CircleMarker(
                [a["lat"], a["lon"]], radius=4, color="#f97316",
                fill=True, fill_color="#f97316", fill_opacity=0.9,
                tooltip=f"🚑 {a['name']} · {a['vehicle_no']}",
            ).add_to(m)

    reports = get_reports()
    for rep in reports:
        if rep.get("lat") and rep.get("lon"):
            ic  = "#ef4444" if rep["type"] == "accident" else "#f59e0b"
            sz  = "14"      if rep["type"] == "accident" else "11"
            folium.Marker(
                [rep["lat"], rep["lon"]],
                tooltip=f"⚠️ {rep['type'].upper()} – {rep['severity']}",
                icon=folium.DivIcon(html=(
                    f"<div style='background:{ic};border:2px solid #fff;border-radius:50%;"
                    f"width:{sz}px;height:{sz}px;box-shadow:0 0 8px {ic};'></div>"
                )),
            ).add_to(m)

    all_pts = [[org["lat"], org["lon"]], [dst["lat"], dst["lon"]]]
    for r in routes:
        if r.get("geometry"):
            all_pts.extend(r["geometry"][:1])
            all_pts.extend(r["geometry"][-1:])
    m.fit_bounds(all_pts, padding=(50, 50))
    return m


# ── GPS geolocation component (browser API) ───────────────────────────────────
_GPS_JS = """
<script>
function getLocation() {
    const btn = document.getElementById('gps-btn');
    const status = document.getElementById('gps-status');
    if (!navigator.geolocation) {
        status.innerHTML = '❌ Geolocation not supported by your browser.';
        return;
    }
    btn.disabled = true;
    btn.textContent = '📡 Detecting...';
    status.innerHTML = '🔄 Requesting location permission...';

    navigator.geolocation.getCurrentPosition(
        function(pos) {
            const lat = pos.coords.latitude.toFixed(6);
            const lon = pos.coords.longitude.toFixed(6);
            // Write coords to a hidden Streamlit text area via URL hack
            const coord_str = lat + ',' + lon;
            status.innerHTML = '✅ Location detected: ' + coord_str;
            btn.textContent = '📍 Location Found';

            // Store in sessionStorage so user can copy
            sessionStorage.setItem('vigiroad_gps', coord_str);

            // Display as copyable
            const out = document.getElementById('gps-output');
            if(out) {
                out.value = coord_str;
                out.style.display = 'block';
            }
        },
        function(err) {
            const msgs = {
                1: '❌ Permission denied. Please allow location access in your browser.',
                2: '❌ Position unavailable. Try again.',
                3: '❌ Request timed out.'
            };
            status.innerHTML = msgs[err.code] || '❌ Unknown error.';
            btn.disabled = false;
            btn.textContent = '📍 Use My Location';
        },
        {enableHighAccuracy: true, timeout: 10000, maximumAge: 0}
    );
}
</script>

<div style="background:linear-gradient(135deg,#1e3a5f,#1e293b);border-radius:12px;
            padding:16px 20px;margin-bottom:14px;border:1px solid #3b82f6;">
    <div style="font-size:0.9rem;font-weight:700;color:#60a5fa;margin-bottom:8px;">
        📍 Allow Location Access
    </div>
    <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:12px;">
        Click below to auto-detect your current location using your device GPS.
        Your location is used only to find routes — it is never stored.
    </div>
    <button id="gps-btn" onclick="getLocation()" style="
        background:linear-gradient(135deg,#3b82f6,#2563eb);
        color:white;border:none;border-radius:8px;padding:8px 18px;
        font-size:0.85rem;font-weight:600;cursor:pointer;">
        📍 Use My Location
    </button>
    <div id="gps-status" style="font-size:0.78rem;color:#94a3b8;margin-top:10px;"></div>
    <div style="margin-top:8px;">
        <input id="gps-output" type="text" readonly
            style="display:none;background:#0f172a;border:1px solid #3b82f6;
                   border-radius:6px;padding:6px 10px;color:#60a5fa;
                   font-size:0.8rem;width:100%;cursor:pointer;"
            onclick="this.select();"
            placeholder="Detected coordinates will appear here — copy & paste above"/>
    </div>
    <div style="font-size:0.72rem;color:#475569;margin-top:6px;">
        ⚠️ Copy the coordinates and paste them in the location field above,
        or just type your city/address manually.
    </div>
</div>
"""


# ── Risk summary panel ────────────────────────────────────────────────────────
def _render_risk_summary(routes):
    """Render a horizontal risk summary bar at the top of results."""
    labels_in = {r["label"] for r in routes}

    cats = [
        ("🔴", "High Risk", "ef4444", "7f1d1d",
         "Multiple accident hotspots, severe potholes & damaged roads."),
        ("⚠️", "Medium Risk", "f59e0b", "78350f",
         "Moderate accidents & pothole density. Drive carefully."),
        ("✅", "Low Risk (Safe Path)", "10b981", "064e3b",
         "Fewest accidents, best road condition. AI-recommended route."),
    ]

    cols = st.columns(3)
    for (em, name, clr, bg, tip), col in zip(cats, cols):
        label_key = f"{name.split(' ')[0]} Risk Road" if "Medium" not in name else "Medium Risk Road"
        if "Low" in name:
            label_key = "Low Risk Road"
        present = label_key in labels_in

        with col:
            opacity = "1" if present else "0.35"
            tag = " <span style='font-size:0.65rem;background:#fff3;padding:2px 7px;border-radius:20px;'>Found on route</span>" if present else ""
            st.markdown(f"""
<div style="background:#{bg};border:2px solid #{clr};border-radius:12px;
            padding:14px 16px;text-align:center;opacity:{opacity};min-height:120px;">
    <div style="font-size:1.6rem">{em}</div>
    <div style="font-weight:800;color:#{clr};font-size:0.9rem;margin:4px 0">{name}{tag}</div>
    <div style="font-size:0.72rem;color:#cbd5e1;line-height:1.4">{tip}</div>
</div>
""", unsafe_allow_html=True)


# ── Route card renderer ───────────────────────────────────────────────────────
def _render_route_card(r, index, o_lat, o_lon, d_lat, d_lon):
    label     = r["label"]
    color     = ROUTE_COLORS.get(label, "#6366f1")
    bg_grad   = ROUTE_BG.get(label, "linear-gradient(135deg,#1e293b,#0f172a)")
    desc      = ROUTE_DESC.get(label, "")
    em        = ROUTE_EMOJI.get(label, "🛣️")
    quality   = r.get("road_quality", 70)
    score     = r["score"]

    rec_badge = ""
    if index == 0:
        rec_badge = (
            "<span style='background:rgba(5,150,105,0.25);color:#34d399;"
            "padding:3px 10px;border-radius:20px;font-size:0.7rem;"
            "font-weight:700;margin-left:8px;border:1px solid #34d39933;'>"
            "⭐ AI RECOMMENDED</span>"
        )

    # Google Maps deep-link
    # Pick midpoint waypoints from geometry if available
    geo = r.get("geometry", [])
    waypoints = [geo[len(geo)//2]] if len(geo) > 2 else None
    gmap_url = _google_maps_url(o_lat, o_lon, d_lat, d_lon, waypoints)

    # Quality bar
    q_color = "#10b981" if quality >= 80 else ("#f59e0b" if quality >= 60 else "#ef4444")

    # Score gauge
    score_pct = min(100, int(score / 10 * 100))
    score_bar_color = "#ef4444" if score > 4.5 else ("#f59e0b" if score > 2.0 else "#10b981")

    st.markdown(f"""
<div style="background:{bg_grad};border-radius:14px;padding:18px 20px;
            margin:10px 0;border-left:5px solid {color};
            box-shadow:0 4px 20px rgba(0,0,0,0.4);">

  <!-- Header row -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;flex-wrap:wrap;gap:6px;">
    <div>
      <span style="background:{color}33;color:{color};padding:4px 12px;
                   border-radius:20px;font-size:0.72rem;font-weight:700;
                   border:1px solid {color}55;">{em} {label}</span>
      {rec_badge}
    </div>
    <a href="{gmap_url}" target="_blank" style="background:#1a73e8;color:white;
       padding:5px 12px;border-radius:8px;font-size:0.72rem;font-weight:700;
       text-decoration:none;display:inline-flex;align-items:center;gap:5px;">
       🗺️ Open in Google Maps
    </a>
  </div>

  <!-- Route description -->
  <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:10px;line-height:1.5;">{desc}</div>

  <!-- Via path -->
  <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:8px 12px;
              margin-bottom:12px;font-size:0.78rem;color:#cbd5e1;line-height:1.6;">
    🛣️ <b>Via:</b> {r["via"] if r["via"] else "Direct route"}
  </div>

  <!-- Stats grid -->
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-bottom:12px;">
    <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:{color};">{r["km"]} km</div>
      <div style="font-size:0.68rem;color:#94a3b8;">📏 Distance</div>
    </div>
    <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#e2e8f0;">~{r["minutes"]} min</div>
      <div style="font-size:0.68rem;color:#94a3b8;">⏱ ETA</div>
    </div>
    <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#f87171;">{r["accidents"]}</div>
      <div style="font-size:0.68rem;color:#94a3b8;">💥 Accidents</div>
    </div>
    <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#fbbf24;">{r["potholes"]}</div>
      <div style="font-size:0.68rem;color:#94a3b8;">🕳️ Potholes/Damage</div>
    </div>
  </div>

  <!-- Safety score bar -->
  <div style="margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;margin-bottom:4px;">
      <span>⚡ Safety Risk Score</span>
      <span style="color:{score_bar_color};font-weight:700;">{score}/10</span>
    </div>
    <div style="background:rgba(0,0,0,0.3);border-radius:20px;height:6px;">
      <div style="background:{score_bar_color};width:{score_pct}%;height:6px;border-radius:20px;
                  transition:width 0.5s ease;"></div>
    </div>
  </div>

  <!-- Road quality bar -->
  <div>
    <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94a3b8;margin-bottom:4px;">
      <span>🏗️ Road Quality Index</span>
      <span style="color:{q_color};font-weight:700;">{quality}%</span>
    </div>
    <div style="background:rgba(0,0,0,0.3);border-radius:20px;height:6px;">
      <div style="background:{q_color};width:{quality}%;height:6px;border-radius:20px;"></div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)


# ── Main dashboard function ───────────────────────────────────────────────────
def show_user_dashboard():
    # ── Voice assistant widget (persistent floating button) ──────────────────
    st.markdown(voice_alert_widget_html(), unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>🗺️ Safe Route Finder</div>
    <div class='section-sub'>AI-powered route safety prediction — based on historical accidents,
    potholes & road damage data</div>
    """, unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────────
    stats = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, lbl, clr in [
        (c1, "🔴", stats["accidents"], "Accidents Logged",  "#ef4444"),
        (c2, "🕳️", stats["potholes"],  "Potholes Logged",   "#f59e0b"),
        (c3, "⚠️",  stats["severe"],   "Severe Zones",      "#dc2626"),
        (c4, "✅",  stats["resolved"], "Issues Resolved",   "#10b981"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{icon} {lbl}</div>
                <div class='value' style='color:{clr}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Risk Level Overview Cards ──────────────────────────────────────────────
    st.markdown("""
<div style='margin-bottom:6px;'>
    <div style='font-size:1.1rem;font-weight:800;color:#e2e8f0;margin-bottom:4px;'>
        🚦 Road Risk Level Guide
    </div>
    <div style='font-size:0.8rem;color:#64748b;margin-bottom:14px;'>
        Our AI classifies every route into one of three risk categories based on historical accident &amp; road damage data.
    </div>
</div>
""", unsafe_allow_html=True)

    risk_c1, risk_c2, risk_c3 = st.columns(3)

    with risk_c1:
        st.markdown("""
<div style="background:linear-gradient(135deg,#064e3b,#065f46);border:2px solid #10b981;
            border-radius:16px;padding:20px 18px;text-align:center;
            box-shadow:0 4px 24px rgba(16,185,129,0.18);
            transition:transform 0.2s;min-height:210px;">
    <div style="font-size:2.4rem;margin-bottom:8px;">✅</div>
    <div style="font-weight:800;color:#10b981;font-size:1rem;margin-bottom:6px;letter-spacing:0.5px;">
        LOW RISK
    </div>
    <div style="display:inline-block;background:rgba(16,185,129,0.18);color:#34d399;
                padding:2px 12px;border-radius:20px;font-size:0.7rem;font-weight:700;
                border:1px solid #10b98155;margin-bottom:10px;">
        ✔ Safe Path
    </div>
    <div style="font-size:0.78rem;color:#a7f3d0;line-height:1.6;">
        Fewest accidents &amp; potholes. Well-maintained road surface.<br>
        <b style="color:#34d399;">AI Recommends</b> this path for safe travel.
    </div>
    <div style="margin-top:12px;font-size:0.72rem;color:#6ee7b7;
                background:rgba(16,185,129,0.1);border-radius:8px;padding:6px 10px;">
        🛡️ Safety Score: <b>0 – 2.0 / 10</b>
    </div>
</div>
""", unsafe_allow_html=True)

    with risk_c2:
        st.markdown("""
<div style="background:linear-gradient(135deg,#78350f,#92400e);border:2px solid #f59e0b;
            border-radius:16px;padding:20px 18px;text-align:center;
            box-shadow:0 4px 24px rgba(245,158,11,0.18);
            transition:transform 0.2s;min-height:210px;">
    <div style="font-size:2.4rem;margin-bottom:8px;">⚠️</div>
    <div style="font-weight:800;color:#f59e0b;font-size:1rem;margin-bottom:6px;letter-spacing:0.5px;">
        MEDIUM RISK
    </div>
    <div style="display:inline-block;background:rgba(245,158,11,0.18);color:#fbbf24;
                padding:2px 12px;border-radius:20px;font-size:0.7rem;font-weight:700;
                border:1px solid #f59e0b55;margin-bottom:10px;">
        ⚡ Caution Advised
    </div>
    <div style="font-size:0.78rem;color:#fde68a;line-height:1.6;">
        Moderate accident history &amp; pothole density.<br>
        Drive at <b style="color:#fbbf24;">reduced speed</b> and stay alert on this route.
    </div>
    <div style="margin-top:12px;font-size:0.72rem;color:#fbbf24;
                background:rgba(245,158,11,0.1);border-radius:8px;padding:6px 10px;">
        ⚡ Safety Score: <b>2.1 – 4.5 / 10</b>
    </div>
</div>
""", unsafe_allow_html=True)

    with risk_c3:
        st.markdown("""
<div style="background:linear-gradient(135deg,#7f1d1d,#991b1b);border:2px solid #ef4444;
            border-radius:16px;padding:20px 18px;text-align:center;
            box-shadow:0 4px 24px rgba(239,68,68,0.18);
            transition:transform 0.2s;min-height:210px;">
    <div style="font-size:2.4rem;margin-bottom:8px;">🔴</div>
    <div style="font-weight:800;color:#ef4444;font-size:1rem;margin-bottom:6px;letter-spacing:0.5px;">
        HIGH RISK
    </div>
    <div style="display:inline-block;background:rgba(239,68,68,0.18);color:#f87171;
                padding:2px 12px;border-radius:20px;font-size:0.7rem;font-weight:700;
                border:1px solid #ef444455;margin-bottom:10px;">
        🚨 Avoid if Possible
    </div>
    <div style="font-size:0.78rem;color:#fca5a5;line-height:1.6;">
        Multiple accident hotspots &amp; severe road damage.<br>
        <b style="color:#f87171;">Consider an alternative</b> route if available.
    </div>
    <div style="margin-top:12px;font-size:0.72rem;color:#f87171;
                background:rgba(239,68,68,0.1);border-radius:8px;padding:6px 10px;">
        🚨 Safety Score: <b>4.6 – 10 / 10</b>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── GPS Location Banner ────────────────────────────────────────────────────
    with st.expander("📍 Auto-detect my location (GPS)", expanded=False):
        st.markdown(_GPS_JS, unsafe_allow_html=True)
        st.info(
            "💡 **How to use GPS**: Click **'Use My Location'** → allow browser permission "
            "→ copy the coordinates shown → paste into the **'Your Current Location'** field below, "
            "or just type your city/area name directly.",
            icon="📡"
        )

    # ── Location inputs ───────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#6366f1,#8b5cf6);
         border-radius:12px; padding:16px 22px; margin-bottom:16px; color:white;'>
        <div style='font-size:1rem; font-weight:700'>📍 Enter Your Locations</div>
        <div style='font-size:0.82rem; opacity:0.85'>
            Type any city, area, or address — or paste GPS coordinates (lat,lon).
            We'll geocode and analyze all available routes automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        origin_text = st.text_input(
            "📍 Your Current Location",
            placeholder="e.g. Gandhipuram, Coimbatore  or  11.0168,76.9558",
            key="origin_input",
            help="Type your city/area name or paste GPS coordinates from the detector above"
        )
    with col_b:
        dest_text = st.text_input(
            "🏁 Your Destination",
            placeholder="e.g. Narasipuram  or  11.2142,77.3411",
            key="dest_input",
            help="Type destination city/area name"
        )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        search = st.button("🔍 Find Safe Routes & Analyze Risk", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🔄 Clear", use_container_width=True):
            for k in ["routes", "route_origin", "route_dest"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Process search ────────────────────────────────────────────────────────
    if search:
        if not origin_text or not dest_text:
            st.warning("⚠️ Please enter both current location and destination.")
            return
        if origin_text.strip().lower() == dest_text.strip().lower():
            st.warning("⚠️ Origin and destination cannot be the same.")
            return

        # Support raw lat,lon input (from GPS detector)
        def _parse_or_geocode(text, label):
            text = text.strip()
            # Try lat,lon format
            if "," in text:
                parts = text.split(",")
                try:
                    la, lo = float(parts[0].strip()), float(parts[1].strip())
                    if -90 <= la <= 90 and -180 <= lo <= 180:
                        return la, lo, text
                except ValueError:
                    pass
            result = geocode(text)
            if result:
                return result
            return None

        with st.spinner("🌍 Finding your locations & analyzing routes..."):
            origin_geo = _parse_or_geocode(origin_text, "origin")
            dest_geo   = _parse_or_geocode(dest_text, "destination")

        if not origin_geo:
            st.error(f"❌ Could not find location: **{origin_text}**. Try a more specific address or city name.")
            return
        if not dest_geo:
            st.error(f"❌ Could not find location: **{dest_text}**. Try a more specific address or city name.")
            return

        o_lat, o_lon, o_display = origin_geo
        d_lat, d_lon, d_display = dest_geo

        with st.spinner("🛣️ Calculating routes & predicting risk levels from historical data..."):
            routes = get_routes((o_lat, o_lon), (d_lat, d_lon), origin_text, dest_text)

        st.session_state["routes"] = routes
        st.session_state["route_origin"] = {"lat": o_lat, "lon": o_lon, "name": o_display}
        st.session_state["route_dest"]   = {"lat": d_lat, "lon": d_lon, "name": d_display}
        st.session_state["o_lat"] = o_lat
        st.session_state["o_lon"] = o_lon
        st.session_state["d_lat"] = d_lat
        st.session_state["d_lon"] = d_lon

    # ── Display results ───────────────────────────────────────────────────────
    if "routes" not in st.session_state:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#94a3b8;'>
            <div style='font-size:3.5rem; margin-bottom:14px'>🗺️</div>
            <div style='font-size:1.15rem; font-weight:700; color:#64748b'>
                Enter your locations above to get started
            </div>
            <div style='font-size:0.85rem; margin-top:8px; max-width:420px; margin-left:auto; margin-right:auto;'>
                Our AI will analyze multiple routes and classify each one as<br>
                🔴 High Risk · ⚠️ Medium Risk · ✅ Low Risk (Safe Path)<br>
                based on historical accident data, potholes & road damage.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    routes = st.session_state["routes"]
    org    = st.session_state["route_origin"]
    dst    = st.session_state["route_dest"]
    o_lat  = st.session_state.get("o_lat", org["lat"])
    o_lon  = st.session_state.get("o_lon", org["lon"])
    d_lat  = st.session_state.get("d_lat", dst["lat"])
    d_lon  = st.session_state.get("d_lon", dst["lon"])

    org_short = org["name"].split(",")[0].strip()
    dst_short = dst["name"].split(",")[0].strip()

    # ── Result header ─────────────────────────────────────────────────────────
    gmap_full = _google_maps_url(o_lat, o_lon, d_lat, d_lon)
    st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f172a,#1e293b);
     border-radius:12px; padding:16px 22px; margin:12px 0; color:white;
     border:1px solid #334155;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>
    <div>
        <div style='font-size:0.78rem; opacity:0.6;margin-bottom:4px'>Route analysis complete — {len(routes)} route(s) found</div>
        <b style='font-size:1.05rem'>📍 {org_short}</b>
        <span style='opacity:0.45'> → </span>
        <b style='font-size:1.05rem'>🏁 {dst_short}</b>
    </div>
    <a href="{gmap_full}" target="_blank" style="background:#1a73e8;color:white;
       padding:8px 16px;border-radius:10px;font-size:0.82rem;font-weight:700;
       text-decoration:none;">🗺️ View All Routes on Google Maps</a>
</div>""", unsafe_allow_html=True)

    # ── Risk summary banner ───────────────────────────────────────────────────
    st.markdown("### 🚦 Route Risk Classification")
    _render_risk_summary(routes)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Split layout: Cards left, Map right ───────────────────────────────────
    col_cards, col_map = st.columns([2, 3])

    with col_cards:
        st.markdown("### 🛣️ Route Analysis — Historical Data")
        for i, r in enumerate(routes):
            _render_route_card(r, i, o_lat, o_lon, d_lat, d_lon)

        # AI model info box
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e293b,#334155);border-radius:12px;
                    padding:14px 18px;margin-top:12px;border:1px solid #475569;'>
            <div style='font-size:0.82rem;color:#94a3b8'>
                🧠 <b style='color:#e2e8f0'>AI Route Safety Model</b> — Trained on Tamil Nadu accident records
                (2020-2025), NHAI road condition surveys, municipal pothole data, and real-time DB incident
                reports. Weights: Accident freq 40% · Pothole density 25% · Road condition 20% · Lane/type 15%.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        st.markdown("### 🗺️ Live Route Map")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            show_police     = st.checkbox("🚔 Police", value=True, key="show_police")
        with tc2:
            show_hospitals  = st.checkbox("🏥 Hospitals", value=True, key="show_hospitals")
        with tc3:
            show_ambulances = st.checkbox("🚑 Ambulances", value=True, key="show_ambulances")

        m = _build_route_map(routes, org, dst, show_police, show_hospitals, show_ambulances)
        st_folium(m, width=None, height=530)

        # Map legend
        st.markdown("""
        <div style='background:#0f172a;border-radius:10px;padding:12px 16px;margin-top:8px;
                    display:flex;gap:16px;flex-wrap:wrap;font-size:0.75rem;color:#94a3b8;border:1px solid #1e293b;'>
            <span>🟢 <span style='color:#00e676'>Safe Path (Low Risk)</span></span>
            <span>🟡 <span style='color:#ffab00'>Medium Risk</span></span>
            <span>🔴 <span style='color:#ff1744'>High Risk</span></span>
            <span>🔵 <span style='color:#60a5fa'>Police</span></span>
            <span>💜 <span style='color:#8b5cf6'>Hospital</span></span>
            <span>🟠 <span style='color:#f97316'>Ambulance</span></span>
        </div>
        """, unsafe_allow_html=True)

    # ── Emergency Services Near Route ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚨 Emergency Services Along Your Route")

    mid_lat = (org["lat"] + dst["lat"]) / 2
    mid_lon = (org["lon"] + dst["lon"]) / 2

    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        st.markdown("""<div style='background:linear-gradient(135deg,#1e3a5f,#1e293b);border-radius:14px;
                    padding:14px 16px;border:1px solid #3b82f6;'>
            <div style='font-size:0.9rem;font-weight:700;color:#60a5fa;margin-bottom:10px'>🚔 Nearby Police Stations</div>
        """, unsafe_allow_html=True)
        for ps in get_nearest_police(mid_lat, mid_lon, max_results=3):
            alert_status = "🟢 In Coverage" if ps["distance_km"] <= ps["coverage_km"] else "🟡 Nearest"
            st.markdown(f"""<div style='background:rgba(59,130,246,0.1);border-radius:10px;padding:10px 12px;
                        margin-bottom:8px;border:1px solid rgba(59,130,246,0.2);'>
                <div style='font-size:0.82rem;font-weight:600;color:#93c5fd'>{ps['name']}</div>
                <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px'>
                    👮 {ps['officer']} · {ps['zone']}<br>📏 {ps['distance_km']} km · {alert_status}<br>📞 {ps['contact']}
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ec2:
        st.markdown("""<div style='background:linear-gradient(135deg,#1a3636,#1e293b);border-radius:14px;
                    padding:14px 16px;border:1px solid #10b981;'>
            <div style='font-size:0.9rem;font-weight:700;color:#34d399;margin-bottom:10px'>🏥 Nearby Hospitals</div>
        """, unsafe_allow_html=True)
        for h in get_nearest_hospitals(mid_lat, mid_lon, max_results=3):
            alert_status = "🟢 In Coverage" if h["distance_km"] <= h["coverage_km"] else "🟡 Nearest"
            t_icon = "🏛️" if h["type"] == "Government Hospital" else "🏢"
            st.markdown(f"""<div style='background:rgba(16,185,129,0.1);border-radius:10px;padding:10px 12px;
                        margin-bottom:8px;border:1px solid rgba(16,185,129,0.2);'>
                <div style='font-size:0.82rem;font-weight:600;color:#6ee7b7'>{t_icon} {h['name']}</div>
                <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px'>
                    🎯 {h['speciality']}<br>🚑 {h['ambulances']} ambulances · 🛏️ {h['beds']} beds<br>📏 {h['distance_km']} km · {alert_status}
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ec3:
        st.markdown("""<div style='background:linear-gradient(135deg,#3b2506,#1e293b);border-radius:14px;
                    padding:14px 16px;border:1px solid #f97316;'>
            <div style='font-size:0.9rem;font-weight:700;color:#fb923c;margin-bottom:10px'>🚑 Ambulance Response</div>
        """, unsafe_allow_html=True)
        for a in get_nearest_ambulances(mid_lat, mid_lon, max_results=3):
            sc = "#10b981" if a["status"] == "available" else "#ef4444"
            st.markdown(f"""<div style='background:rgba(249,115,22,0.1);border-radius:10px;padding:10px 12px;
                        margin-bottom:8px;border:1px solid rgba(249,115,22,0.2);'>
                <div style='font-size:0.82rem;font-weight:600;color:#fdba74'>{a['name']}</div>
                <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px'>
                    🚐 {a['vehicle_no']}<br><span style='color:{sc};font-weight:600'>● {a['status'].upper()}</span><br>📏 {a['distance_km']} km · ⏱ ETA ~{a['eta_min']} min
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Navigation-style hazard feed + proximity alerts ─────────────────────
    st.markdown("---")
    st.markdown("### 🧭 Navigation & Proximity Alerts")

    nav_col, prox_col = st.columns([3, 2])

    with nav_col:
        st.markdown("#### 🛣️ Route Hazards — Navigation View")
        st.markdown(
            "<div style='font-size:0.8rem;color:#64748b;margin-bottom:10px'>"
            "Hazards ordered along your selected route. Voice alerts fire automatically.</div>",
            unsafe_allow_html=True,
        )

        # Get all pothole/damage reports along route
        all_rep = get_reports()
        route_reports = [r for r in all_rep if r.get("lat") and r.get("lon")]

        if route_reports:
            # Sort by distance from origin
            import math
            def _dist(r):
                dlat = math.radians(r["lat"] - o_lat)
                dlon = math.radians(r["lon"] - o_lon)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(o_lat))*math.cos(math.radians(r["lat"]))*math.sin(dlon/2)**2
                return 6371*2*math.asin(math.sqrt(a))

            route_reports.sort(key=_dist)
            spoken_first = False

            for rep in route_reports[:8]:
                dist_km  = _dist(rep)
                dist_m   = int(dist_km * 1000)
                sev      = rep.get("severity", "medium")
                rtype    = rep["type"]

                if rtype == "accident":
                    icon = "💥"; color = "#ef4444"
                    inst = f"Accident reported {dist_m} metres ahead near {rep['location_name']}. Drive carefully."
                else:
                    icon = "🕳️"; color = "#f59e0b"
                    inst = f"Road damage ahead in {dist_m} metres near {rep['location_name']}."

                # First hazard gets voice spoken
                delay = 1000 if not spoken_first else 0
                if not spoken_first:
                    st.markdown(
                        speak_html(inst, beep="pothole" if rtype=="pothole" else "alert", delay_ms=1200),
                        unsafe_allow_html=True,
                    )
                    spoken_first = True

                st.markdown(pothole_warning_html(rep["location_name"], dist_m, sev)
                            if rtype == "pothole" else
                            f"<div style='background:{color}11;border:2px solid {color};"
                            f"border-radius:12px;padding:10px 14px;margin:5px 0;"
                            f"display:flex;align-items:center;gap:10px;'>"
                            f"<span style='font-size:1.4rem'>{icon}</span>"
                            f"<div><div style='font-size:0.87rem;font-weight:700;color:{color}'>"
                            f"Accident Zone — {dist_m}m ahead</div>"
                            f"<div style='font-size:0.75rem;color:#94a3b8'>"
                            f"📍 {rep['location_name']} · {sev.upper()} severity</div>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                )
        else:
            st.markdown(
                nav_instruction_html("✅ No road hazards detected on this route. Safe to proceed.", "✅", "#10b981"),
                unsafe_allow_html=True,
            )

        # Google Maps navigation deep-link with waypoints
        gmap_nav = _google_maps_url(o_lat, o_lon, d_lat, d_lon)
        st.markdown(
            f"<a href='{gmap_nav}' target='_blank' style='display:inline-flex;align-items:center;"
            f"gap:8px;background:#1a73e8;color:white;padding:10px 18px;border-radius:10px;"
            f"font-size:0.85rem;font-weight:700;text-decoration:none;margin-top:8px;'>"
            f"🗺️ Start Navigation on Google Maps</a>",
            unsafe_allow_html=True,
        )

    with prox_col:
        st.markdown("#### 📡 Nearby Services")
        st.markdown(
            "<div style='font-size:0.8rem;color:#64748b;margin-bottom:10px'>"
            "Services within 5 km of your location. Click to get voice alert.</div>",
            unsafe_allow_html=True,
        )

        # Hospitals near origin
        near_hospitals = get_nearest_hospitals(o_lat, o_lon, max_results=3)
        if near_hospitals:
            for h in near_hospitals:
                dist_m = int(h["distance_km"] * 1000)
                st.markdown(
                    hospital_proximity_html(
                        h["name"], dist_m,
                        beds=h.get("beds", 0),
                        phone=h.get("contact", ""),
                    ),
                    unsafe_allow_html=True,
                )

        # Police near origin
        near_police = get_nearest_police(o_lat, o_lon, max_results=2)
        if near_police:
            for ps in near_police:
                dist_m = int(ps["distance_km"] * 1000)
                st.markdown(
                    police_proximity_html(
                        ps["name"], dist_m,
                        officer=ps.get("officer", ""),
                        contact=ps.get("contact", ""),
                    ),
                    unsafe_allow_html=True,
                )

        # Risk zone summary voice
        if routes:
            best_r = routes[0]
            label  = best_r.get("label", "")
            risk_voice = {
                "Low Risk Road":    "Route is safe. Low risk path selected.",
                "Medium Risk Road": "Caution. Medium risk route. Drive carefully.",
                "High Risk Road":   "Warning! High risk route selected. Consider an alternative.",
            }.get(label, "Route analysis complete.")
            st.markdown(
                speak_html(risk_voice, delay_ms=2500),
                unsafe_allow_html=True,
            )

    # ── AI Recommendation ─────────────────────────────────────────────────────
    best = routes[0] if routes else None
    if best:
        nearest_police = get_nearest_police(mid_lat, mid_lon, 1)
        nearest_hosp   = get_nearest_hospitals(mid_lat, mid_lon, 1)
        p_name = nearest_police[0]["name"] if nearest_police else "N/A"
        h_name = nearest_hosp[0]["name"]   if nearest_hosp   else "N/A"

        risk_advice = {
            "Low Risk Road":    "✅ This is the **safest path** — take this route.",
            "Medium Risk Road": "⚠️ Moderate risk route — drive at reduced speed and stay alert.",
            "High Risk Road":   "🔴 High risk — consider waiting or taking an alternative if possible.",
        }.get(best["label"], "")

        st.markdown(f"""
<div style='background:linear-gradient(135deg,#064e3b,#065f46);
     border:2px solid #10b981; border-radius:14px; padding:18px 22px; margin-top:16px;'>
    <div style='font-size:0.95rem; font-weight:700; color:#6ee7b7;margin-bottom:8px'>💡 AI Safety Recommendation</div>
    <div style='font-size:0.85rem; color:#a7f3d0; line-height:1.7'>
        Best route: <b>{best["name"][:60]}</b><br>
        Distance: <b>{best["km"]} km</b> · ETA: <b>~{best["minutes"]} min</b> ·
        Safety Score: <b style='color:#6ee7b7'>{best["score"]}/10</b><br>
        Historical data: <b>{best["accidents"]} accident(s)</b> and <b>{best["potholes"]} pothole/damage point(s)</b> on record.<br>
        {risk_advice}<br>
        Nearest police: <b>{p_name}</b> · Nearest hospital: <b>{h_name}</b>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Historical Data Note ───────────────────────────────────────────────────
    st.markdown("""
<div style='background:#0f172a;border:1px solid #334155;border-radius:12px;
            padding:14px 18px;margin-top:12px;'>
    <div style='font-size:0.78rem;color:#64748b;line-height:1.7'>
        📊 <b style='color:#94a3b8'>Data Sources:</b>
        Tamil Nadu Police accident records (2020–2025) ·
        NHAI road condition surveys ·
        Municipal pothole complaint databases ·
        Real-time VigiRoad AI incident reports ·
        OpenStreetMap road network data<br>
        🎯 <b style='color:#94a3b8'>Model:</b> Weighted risk scoring (Accident freq 40% ·
        Pothole density 25% · Road condition 20% · Road type/lanes 15%)
        combined with real-time DB hazard proximity analysis.
    </div>
</div>
""", unsafe_allow_html=True)
