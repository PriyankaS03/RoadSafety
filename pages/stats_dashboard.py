"""
pages/stats_dashboard.py – Analytics dashboard for all roles
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.db import get_reports, get_stats

def show_stats_dashboard():
    st.markdown("""
    <div class='section-header'>📊 Road Safety Analytics</div>
    <div class='section-sub'>Tamil Nadu – comprehensive hazard statistics & trends</div>
    """, unsafe_allow_html=True)

    stats = get_stats()
    cols = st.columns(6)
    for col, icon, val, lbl, clr in [
        (cols[0], "📋", stats["total"],     "Total Reports",   "#6366f1"),
        (cols[1], "💥", stats["accidents"], "Accidents",       "#ef4444"),
        (cols[2], "🕳️", stats["potholes"],  "Potholes",        "#f59e0b"),
        (cols[3], "⚠️", stats["severe"],   "Severe",          "#dc2626"),
        (cols[4], "⏳", stats["pending"],   "Pending",         "#f97316"),
        (cols[5], "✅", stats["resolved"],  "Resolved",        "#10b981"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card' style='text-align:center'>
                <div class='label'>{icon}<br>{lbl}</div>
                <div class='value' style='color:{clr};font-size:1.8rem'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    all_reports = get_reports()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Hazard Type Distribution")
        acc_c = sum(1 for r in all_reports if r["type"] == "accident")
        pot_c = sum(1 for r in all_reports if r["type"] == "pothole")
        total = acc_c + pot_c or 1
        for lbl, cnt, clr in [("Accidents", acc_c, "#ef4444"), ("Potholes", pot_c, "#f59e0b")]:
            pct = cnt / total * 100
            st.markdown(f"""
            <div style='margin:10px 0'>
                <div style='display:flex; justify-content:space-between; font-size:0.85rem; color:#475569; margin-bottom:4px'>
                    <span>{lbl}</span><span>{cnt} ({pct:.0f}%)</span>
                </div>
                <div style='background:#e2e8f0; border-radius:8px; height:12px'>
                    <div style='background:{clr}; width:{pct:.0f}%; height:12px; border-radius:8px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Resolution Rate")
        res = stats["resolved"]
        pend = stats["pending"]
        tot2 = res + pend or 1
        res_pct = res / tot2 * 100
        st.markdown(f"""
        <div style='margin:10px 0'>
            <div style='display:flex; justify-content:space-between; font-size:0.85rem; color:#475569; margin-bottom:4px'>
                <span>Resolved</span><span>{res_pct:.0f}%</span>
            </div>
            <div style='background:#e2e8f0; border-radius:8px; height:12px'>
                <div style='background:#10b981; width:{res_pct:.0f}%; height:12px; border-radius:8px'></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### Top Affected Locations")
        loc_cnt = {}
        for r in all_reports:
            loc_cnt[r["location_name"]] = loc_cnt.get(r["location_name"], 0) + 1
        top = sorted(loc_cnt.items(), key=lambda x: x[1], reverse=True)[:8]
        for loc, cnt in top:
            bar_w = int(cnt / max(v for _, v in top) * 100)
            st.markdown(f"""
            <div style='background:white; border-radius:10px; padding:8px 14px; margin:5px 0;
                 box-shadow:0 1px 6px rgba(0,0,0,0.06);'>
                <div style='display:flex; justify-content:space-between; font-size:0.81rem; color:#334155; margin-bottom:4px'>
                    <span>📍 {loc}</span><span style='font-weight:700'>{cnt}</span>
                </div>
                <div style='background:#f1f5f9; border-radius:4px; height:6px'>
                    <div style='background:linear-gradient(90deg,#6366f1,#8b5cf6); width:{bar_w}%; height:6px; border-radius:4px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Full Hazard Map – Tamil Nadu")
    m = folium.Map(location=[10.8, 78.0], zoom_start=7, tiles="CartoDB positron")

    for r in all_reports:
        if r["lat"] and r["lon"]:
            if r["type"] == "accident":
                folium.CircleMarker([r["lat"], r["lon"]], radius=10,
                                    color="#ef4444", fill=True, fill_opacity=0.7,
                                    tooltip=f"💥 Accident – {r['severity']} – {r['location_name']}").add_to(m)
            else:
                clr = {"severe": "#dc2626", "medium": "#f59e0b", "low": "#10b981"}.get(r["severity"], "#94a3b8")
                folium.CircleMarker([r["lat"], r["lon"]], radius=7,
                                    color=clr, fill=True, fill_opacity=0.75,
                                    tooltip=f"🕳️ Pothole – {r['severity']} – {r['location_name']}").add_to(m)

    st_folium(m, width=None, height=450)
