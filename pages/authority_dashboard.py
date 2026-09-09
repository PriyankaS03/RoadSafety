"""
pages/authority_dashboard.py – Road Damage Authority Dashboard
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.db import get_reports, update_report_status, get_stats

SEV_COLOR = {"severe": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}
SEV_BADGE = {"severe": "badge-red", "medium": "badge-amber", "low": "badge-green"}

def show_authority_dashboard():
    st.markdown("""
    <div class='section-header'>🏢 Highways Authority Dashboard</div>
    <div class='section-sub'>Road damage monitoring & maintenance management</div>
    """, unsafe_allow_html=True)

    stats = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, lbl, clr in [
        (c1, "🕳️", stats["potholes"],  "Total Potholes",    "#f59e0b"),
        (c2, "🔴", stats["severe"],    "Severe Damage",     "#ef4444"),
        (c3, "✅", stats["resolved"],  "Repaired",          "#10b981"),
        (c4, "⏳", stats["pending"],   "Pending Repair",    "#dc2626"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>{icon} {lbl}</div>
                <div class='value' style='color:{clr}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Analytics mini-chart
    all_ph = get_reports(type_="pothole")
    # Normalise legacy variants then count only the 3 known keys
    SEV_ALIAS = {"moderate": "medium", "high": "severe", "minor": "low"}
    sev_counts = {"severe": 0, "medium": 0, "low": 0}
    for r in all_ph:
        sev = SEV_ALIAS.get(r["severity"], r["severity"])
        if sev in sev_counts:          # skip any truly unknown value
            sev_counts[sev] += 1

    col_chart, col_info = st.columns([1, 2])
    with col_chart:
        st.markdown("#### Severity Breakdown")
        for sev, cnt in sev_counts.items():   # always exactly 3 known keys
            pct = (cnt / max(len(all_ph), 1)) * 100
            clr = SEV_COLOR.get(sev, "#94a3b8")
            st.markdown(f"""
            <div style='margin:6px 0'>
                <div style='display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; margin-bottom:3px'>
                    <span>{sev.capitalize()}</span><span>{cnt}</span>
                </div>
                <div style='background:#e2e8f0; border-radius:6px; height:8px'>
                    <div style='background:{clr}; width:{pct:.0f}%; height:8px; border-radius:6px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown("#### High-Risk Areas")
        loc_counts = {}
        for r in all_ph:
            loc_counts[r["location_name"]] = loc_counts.get(r["location_name"], 0) + 1
        top5 = sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for loc, cnt in top5:
            st.markdown(f"""
            <div style='background:white; border-radius:10px; padding:10px 14px; margin:5px 0;
                        box-shadow:0 1px 6px rgba(0,0,0,0.07); display:flex; justify-content:space-between;'>
                <span style='font-size:0.84rem; color:#334155'>📍 {loc}</span>
                <span class='badge badge-amber'>{cnt} issues</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Report list
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        filter_sev = st.selectbox("Filter by Severity", ["All", "severe", "medium", "low"])
    with col_f2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    reports = get_reports(type_="pothole")
    if filter_sev != "All":
        reports = [r for r in reports if r["severity"] == filter_sev]

    if not reports:
        st.info("✅ No road damage reports.")
    else:
        st.markdown(f"**{len(reports)} report(s)**")
        for r in reports:
            sb = SEV_BADGE.get(r["severity"], "badge-blue")
            status_clr = "success" if r["status"] == "resolved" else ("warning" if r["severity"] == "medium" else "")
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div class='alert-card {status_clr}'>
                    <div style='display:flex; gap:8px; align-items:center; margin-bottom:6px'>
                        <span>🕳️</span>
                        <b>Report #{r["id"]}</b>
                        <span class='badge {sb}'>{r["severity"].upper()}</span>
                        <span class='badge {"badge-green" if r["status"]=="resolved" else "badge-red"}'>{r["status"].upper()}</span>
                    </div>
                    <div style='font-size:0.83rem; color:#475569'>
                        📍 {r["location_name"]} &nbsp;|&nbsp; 🕐 {r["timestamp"]}
                    </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                if r["status"] == "pending":
                    if st.button("🛠️ Mark Repaired", key=f"auth_{r['id']}", use_container_width=True):
                        update_report_status(r["id"], "resolved")
                        st.toast("Marked as repaired!", icon="🛠️")
                        st.rerun()

    # Map
    st.markdown("### 🗺️ Damage Map")
    m = folium.Map(location=[10.99, 77.01], zoom_start=8, tiles="CartoDB positron")
    for r in get_reports(type_="pothole"):
        if r["lat"] and r["lon"]:
            clr = SEV_COLOR.get(r["severity"], "#94a3b8")
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=8, color=clr, fill=True, fill_opacity=0.8,
                tooltip=f"🕳️ {r['severity'].upper()} – {r['location_name']}"
            ).add_to(m)
    st_folium(m, width=None, height=380)
