"""
pages/police_dashboard.py – Police Command Centre
• Zone-restricted to officer's controlling area
• Voice assistant: beep + speech for every new alert
• Damage-rate threshold: >= 50% = alert card, < 50% = info only
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.db import (
    get_reports_in_area, get_reports, update_report_status,
    get_stats_in_area, alert_level, should_alert,
)
from utils.emergency_services import (
    POLICE_STATIONS, get_nearest_hospitals, get_nearest_police, _haversine,
)
from utils.voice_assistant import (
    voice_alert_widget_html, speak_html,
    pothole_warning_html, hospital_proximity_html,
)

SEV_COLOR    = {"severe": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}
STATUS_COLOR = {"pending": "#ef4444", "resolved": "#10b981", "dispatched": "#3b82f6"}


def _get_user_station():
    user = st.session_state.get("user", {})
    lat, lon = user.get("station_lat"), user.get("station_lon")
    if lat and lon:
        return {
            "lat": lat, "lon": lon,
            "coverage_km": user.get("coverage_km", 10.0),
            "name": user.get("station_name", "Your Station"),
        }
    return None


def _damage_bar_html(rate: int) -> str:
    lvl = alert_level(rate)
    bar_color = "#ef4444" if rate >= 80 else ("#f59e0b" if rate >= 50 else "#10b981")
    return (
        f"<div style='margin-top:6px'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.7rem;"
        f"color:#94a3b8;margin-bottom:2px'>"
        f"<span>💥 Damage Rate</span>"
        f"<span style='color:{bar_color};font-weight:700'>{rate}% — {lvl}</span></div>"
        f"<div style='background:#1e293b;border-radius:6px;height:6px;overflow:hidden'>"
        f"<div style='background:{bar_color};width:{rate}%;height:100%;border-radius:6px'></div>"
        f"</div></div>"
    )


def _build_police_map(reports, station):
    center_lat = station["lat"] if station else 11.01
    center_lon = station["lon"] if station else 77.00
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles="CartoDB dark_matter", control_scale=True)

    if station:
        folium.Marker(
            [station["lat"], station["lon"]],
            tooltip=f"🚔 YOUR STATION: {station['name']}",
            icon=folium.DivIcon(html=(
                "<div style='background:#3b82f6;border:4px solid #93c5fd;border-radius:50%;"
                "width:24px;height:24px;display:flex;align-items:center;justify-content:center;"
                "box-shadow:0 0 20px rgba(59,130,246,0.9);'>"
                "<div style='background:#fff;border-radius:50%;width:10px;height:10px;'></div>"
                "</div>"
            )),
        ).add_to(m)
        folium.Circle(
            [station["lat"], station["lon"]],
            radius=station["coverage_km"] * 1000,
            color="#3b82f6", fill=True, fill_color="#3b82f6",
            fill_opacity=0.08, weight=3, dash_array="10 5",
            tooltip=f"🔵 Controlling Area – {station['coverage_km']} km radius",
        ).add_to(m)
        folium.Marker(
            [station["lat"], station["lon"]],
            icon=folium.DivIcon(html=(
                f"<div style='font-size:10px;color:#93c5fd;font-weight:700;white-space:nowrap;"
                f"margin-top:22px;margin-left:-50px;width:120px;text-align:center;"
                f"background:rgba(15,23,42,0.7);padding:2px 6px;border-radius:4px;'>"
                f"🚔 {station['name'][:30]}</div>"
            )),
        ).add_to(m)

    for ps in POLICE_STATIONS:
        if station and abs(ps["lat"] - station["lat"]) < 0.001 and abs(ps["lon"] - station["lon"]) < 0.001:
            continue
        folium.CircleMarker(
            [ps["lat"], ps["lon"]], radius=3, color="#60a5fa55",
            fill=True, fill_color="#3b82f655", fill_opacity=0.4,
            tooltip=f"🚔 {ps['name']}",
        ).add_to(m)

    for r in reports:
        if not r.get("lat") or not r.get("lon"):
            continue
        is_pending  = r["status"] == "pending"
        sev         = r.get("severity", "medium")
        dmg         = r.get("damage_rate", 0)
        is_alerting = should_alert(dmg) and is_pending
        dot_color   = SEV_COLOR.get(sev, "#ef4444") if is_pending else "#6b7280"
        size        = 16 if is_alerting else 12

        icon_html = (
            f"<div style='background:{dot_color};border:2px solid #fff;border-radius:50%;"
            f"width:{size}px;height:{size}px;"
            f"box-shadow:0 0 {12 if is_alerting else 0}px {dot_color};'></div>"
        )
        folium.Marker(
            [r["lat"], r["lon"]],
            tooltip=folium.Tooltip(
                f"<b>💥 Accident #{r['id']}</b><br>"
                f"📍 {r['location_name']}<br>"
                f"🕐 {r['timestamp']}<br>"
                f"💥 Damage: {dmg}% ({alert_level(dmg)})<br>"
                f"Status: {r['status'].upper()}<br>"
                f"📏 {r.get('distance_km','?')} km from station",
                sticky=True,
            ),
            icon=folium.DivIcon(html=icon_html),
        ).add_to(m)

        if station and is_alerting:
            folium.PolyLine(
                [[r["lat"], r["lon"]], [station["lat"], station["lon"]]],
                color="#ef4444", weight=2, opacity=0.5, dash_array="6 4",
            ).add_to(m)

    return m


def show_police_dashboard():
    station = _get_user_station()

    # ── Voice assistant widget ─────────────────────────────────────────────────
    st.markdown(voice_alert_widget_html(), unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>🚨 Police Command Centre</div>
    <div class='section-sub'>Real-time accident alerts within your controlling area</div>
    """, unsafe_allow_html=True)

    if station:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#1e3a5f,#0f172a);border-radius:12px;"
            f"padding:12px 18px;margin-bottom:16px;border:1px solid #3b82f6;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><span style='font-size:0.78rem;color:#94a3b8'>Your Station</span><br>"
            f"<b style='font-size:1rem;color:#93c5fd'>🚔 {station['name']}</b></div>"
            f"<div style='text-align:right'><span style='font-size:0.75rem;color:#94a3b8'>Controlling Area</span><br>"
            f"<b style='font-size:0.9rem;color:#60a5fa'>📡 {station['coverage_km']} km radius</b></div>"
            f"<div style='text-align:right'><span style='font-size:0.75rem;color:#94a3b8'>Coordinates</span><br>"
            f"<span style='font-size:0.82rem;color:#cbd5e1'>{station['lat']:.4f}, {station['lon']:.4f}</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Station location not set. Update your profile.")

    stats = (get_stats_in_area(station["lat"], station["lon"], station["coverage_km"])
             if station else get_stats_in_area())

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, icon, val, lbl, clr in [
        (c1, "💥", stats["accidents"], "Accidents (Area)",  "#ef4444"),
        (c2, "🚨", stats["alerts"],    "Alerts Sent",       "#dc2626"),
        (c3, "🔴", stats["pending"],   "Pending Alerts",    "#f97316"),
        (c4, "✅", stats["resolved"],  "Resolved",          "#10b981"),
        (c5, "⚠️", stats["severe"],   "Severe Incidents",  "#f59e0b"),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='label'>{icon} {lbl}</div>"
                f"<div class='value' style='color:{clr}'>{val}</div>"
                f"</div>", unsafe_allow_html=True,
            )

    st.markdown("---")
    col_alerts, col_map = st.columns([2, 3])

    with col_alerts:
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            filter_status = st.selectbox("Filter", ["All", "pending", "resolved"])
        with col_f2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        status_arg = None if filter_status == "All" else filter_status

        if station:
            reports = get_reports_in_area(
                station["lat"], station["lon"], station["coverage_km"],
                type_="accident", status=status_arg,
            )
        else:
            reports = get_reports(type_="accident", status=status_arg)

        alert_reports = [r for r in reports if should_alert(r.get("damage_rate", 0))]
        low_reports   = [r for r in reports if not should_alert(r.get("damage_rate", 0))]
        pending_alerts = [r for r in alert_reports if r["status"] == "pending"]

        # ── Voice: announce pending alert count ───────────────────────────────
        if pending_alerts:
            n = len(pending_alerts)
            st.markdown(
                speak_html(
                    f"Attention! {n} accident alert{'s' if n > 1 else ''} in your zone "
                    f"requiring immediate action.",
                    beep="alert", delay_ms=800,
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#7f1d1d,#991b1b);"
                f"border:2px solid #ef4444;border-radius:14px;padding:12px 16px;margin:8px 0;'>"
                f"<b style='color:#fca5a5;font-size:0.9rem'>🚨 {n} ALERT(S) SENT TO POLICE</b>"
                f"<div style='color:#fca5a5;font-size:0.74rem;margin-top:3px;opacity:0.85'>"
                f"Damage rate ≥ 50% — Immediate action required</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if not reports:
            st.info("✅ No accident reports in your controlling area.")

        for r in alert_reports:
            sev          = r.get("severity", "medium")
            sev_color    = SEV_COLOR.get(sev, "#f59e0b")
            status_color = STATUS_COLOR.get(r["status"], "#6b7280")
            is_pending   = r["status"] == "pending"
            dmg          = r.get("damage_rate", 0)
            dist_txt     = f" · 📏 {r['distance_km']} km" if r.get("distance_km") else ""
            alert_t      = r.get("alert_time") or r["timestamp"]
            border       = sev_color if is_pending else "#4b5563"

            hosp_html = ""
            if r.get("lat") and r.get("lon"):
                hosp = get_nearest_hospitals(r["lat"], r["lon"], 1)
                if hosp:
                    h0 = hosp[0]
                    hosp_html = (
                        f"<div style='margin-top:5px;padding:6px 10px;"
                        f"background:rgba(16,185,129,0.1);border-radius:8px;"
                        f"font-size:0.72rem;color:#6ee7b7;"
                        f"border:1px solid rgba(16,185,129,0.2)'>"
                        f"🏥 Nearest hospital: {h0['name']} ({h0['distance_km']} km)"
                        f"</div>"
                    )

            st.markdown(
                f"<div style='background:rgba(15,23,42,0.85);border-radius:12px;"
                f"padding:14px 16px;margin:8px 0;border-left:4px solid {border};'>"
                f"<div style='display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:8px'>"
                f"<span style='font-size:1.1rem'>💥</span>"
                f"<b style='font-size:0.92rem;color:#e2e8f0'>Accident #{r['id']}</b>"
                f"<span style='background:{sev_color}22;color:{sev_color};"
                f"padding:2px 8px;border-radius:10px;font-size:0.68rem;font-weight:700'>{sev.upper()}</span>"
                f"<span style='background:{status_color}22;color:{status_color};"
                f"padding:2px 8px;border-radius:10px;font-size:0.68rem;font-weight:700'>{r['status'].upper()}</span>"
                f"<span style='background:#ef444422;color:#ef4444;"
                f"padding:2px 8px;border-radius:10px;font-size:0.68rem;font-weight:700'>🔔 ALERT SENT</span>"
                f"</div>"
                f"<div style='font-size:0.83rem;color:#94a3b8;margin-bottom:3px'>"
                f"📍 <b style='color:#cbd5e1'>{r['location_name']}</b>{dist_txt}</div>"
                f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:2px'>"
                f"🕐 Accident at: <b style='color:#94a3b8'>{r['timestamp']}</b></div>"
                f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:4px'>"
                f"📢 Alert sent at: <b style='color:#f59e0b'>{alert_t}</b></div>"
                f"{_damage_bar_html(dmg)}"
                f"{hosp_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if is_pending:
                if st.button(f"✅ Mark Handled — #{r['id']}", key=f"pol_{r['id']}", use_container_width=True):
                    update_report_status(r["id"], "resolved")
                    st.toast(f"Incident #{r['id']} marked resolved", icon="✅")
                    st.rerun()

        if low_reports:
            with st.expander(f"ℹ️ {len(low_reports)} low-damage incident(s) — no alert"):
                for r in low_reports:
                    dmg      = r.get("damage_rate", 0)
                    dist_txt = f" · 📏 {r['distance_km']} km" if r.get("distance_km") else ""
                    st.markdown(
                        f"<div style='background:rgba(15,23,42,0.5);border-radius:10px;"
                        f"padding:10px 14px;margin:6px 0;border-left:3px solid #10b981;'>"
                        f"<div style='font-size:0.85rem;color:#e2e8f0;font-weight:600'>"
                        f"🕳️ Incident #{r['id']} — <span style='color:#10b981'>LOW DAMAGE ({dmg}%)</span></div>"
                        f"<div style='font-size:0.78rem;color:#94a3b8;margin-top:4px'>"
                        f"📍 {r['location_name']}{dist_txt}</div>"
                        f"<div style='font-size:0.75rem;color:#64748b;margin-top:2px'>"
                        f"🕐 {r['timestamp']} — ℹ️ No alert sent (below 50% threshold)</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    with col_map:
        st.markdown("### 🗺️ Your Jurisdiction Map")
        all_area = (get_reports_in_area(station["lat"], station["lon"],
                                        station["coverage_km"], type_="accident")
                    if station else get_reports(type_="accident"))
        m = _build_police_map(all_area, station)
        st_folium(m, width=None, height=560)
        st.markdown(
            "<div style='background:#0f172a;border-radius:10px;padding:10px 14px;margin-top:8px;"
            "display:flex;gap:14px;flex-wrap:wrap;font-size:0.73rem;color:#94a3b8;'>"
            "<span>🔵 <span style='color:#60a5fa'>Your Station</span></span>"
            "<span style='color:#60a5fa'>◌ Controlling Area</span>"
            "<span>🔴 <span style='color:#ef4444'>Alert Incident</span></span>"
            "<span>🟢 <span style='color:#10b981'>Low Damage</span></span>"
            "<span>⚫ <span style='color:#6b7280'>Resolved</span></span>"
            "</div>",
            unsafe_allow_html=True,
        )
