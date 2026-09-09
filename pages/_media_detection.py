"""
pages/media_detection.py – Image & Video Upload Detection
• Image  → pothole / road damage / road condition detection
• Video  → frame-by-frame road damage + accident detection timeline
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import io
from PIL import Image

from utils.media_detection import (
    mock_analyze_image,
    mock_analyze_video_frame,
    annotate_video_frame,
    build_video_summary,
    DAMAGE_LABELS,
    ACCIDENT_LABELS,
)
from utils.db import add_report
from utils.detection import get_random_tn_location


# ── Colour helpers ────────────────────────────────────────────────────────────

SEVERITY_COLOURS = {
    "Low":      ("#d1fae5", "#059669", "🟢"),
    "Moderate": ("#fef3c7", "#d97706", "🟡"),
    "High":     ("#ffedd5", "#ea580c", "🟠"),
    "Severe":   ("#fee2e2", "#dc2626", "🔴"),
    "Critical": ("#fce7f3", "#9d174d", "🔴"),
}

CONDITION_COLOURS = {
    "Good":      "#059669",
    "Fair":      "#0ea5e9",
    "Poor":      "#f59e0b",
    "Very Poor": "#ef4444",
    "Critical":  "#7c3aed",
}

# ── Shared CSS injected once ───────────────────────────────────────────────────
_CSS = """
<style>
/* ─── Tabs ─────────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-weight: 700; font-size: 0.95rem; padding: 10px 20px; border-radius: 10px 10px 0 0;
}

/* ─── Upload zone ──────────────────────────────────── */
.upload-zone {
    border: 2px dashed #6366f1;
    border-radius: 18px;
    padding: 36px 24px;
    text-align: center;
    background: linear-gradient(135deg,#f8f9ff,#eef0ff);
    margin-bottom: 20px;
}
.upload-zone h3 { color: #4f46e5; margin: 0 0 6px; font-size: 1.1rem; }
.upload-zone p  { color: #64748b; font-size: 0.85rem; margin: 0; }

/* ─── Damage gauge wrapper ─────────────────────────── */
.gauge-card {
    background: white;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    text-align: center;
    border: 1px solid #e2e8f0;
}
.gauge-val  { font-size: 3rem; font-weight: 900; line-height: 1; }
.gauge-lbl  { font-size: 0.78rem; color: #64748b; font-weight: 600;
               text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }
.gauge-cond { font-size: 1rem; font-weight: 700; margin-top: 8px; }

/* ─── Stat pill ────────────────────────────────────── */
.stat-pill {
    display: inline-block;
    background: #f1f5f9;
    border-radius: 30px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #334155;
    margin: 4px 4px;
}

/* ─── Detection item ───────────────────────────────── */
.det-item {
    background: white;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 12px;
    border: 1px solid #f1f5f9;
}
.det-dot  { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.det-name { font-weight: 700; font-size: 0.9rem; color: #0f172a; }
.det-conf { font-size: 0.8rem; color: #64748b; }
.det-sev  { font-size: 0.75rem; font-weight: 700; border-radius: 20px;
             padding: 2px 10px; margin-left: auto; }

/* ─── Accident banner ──────────────────────────────── */
.accident-banner {
    background: linear-gradient(135deg,#7f1d1d,#991b1b);
    border-radius: 16px;
    padding: 20px 24px;
    color: white;
    text-align: center;
    animation: pulse-border 1.5s infinite;
    border: 2px solid #ef4444;
    margin: 12px 0;
}
@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%      { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
}
.accident-banner h2 { font-size: 1.4rem; margin: 0 0 6px; }
.accident-banner p  { font-size: 0.88rem; opacity: 0.85; margin: 0; }

/* ─── Timeline item ────────────────────────────────── */
.timeline-item {
    background: white;
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 10px 16px;
    margin: 6px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.timeline-item.damage { border-left-color: #f59e0b; }
.timeline-item.clean  { border-left-color: #10b981; }

/* ─── Progress bar ─────────────────────────────────── */
.damage-bar-wrap { background: #f1f5f9; border-radius: 99px; height: 10px; margin: 8px 0; overflow: hidden; }
.damage-bar-fill { height: 100%; border-radius: 99px; transition: width 0.4s; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Section: Image Detection
# ─────────────────────────────────────────────────────────────────────────────

def _colour_hex_to_bgr_css(bgr_tuple):
    """Convert OpenCV BGR tuple to CSS rgb() string."""
    b, g, r = bgr_tuple
    return f"rgb({r},{g},{b})"


def _render_image_detections(detections):
    if not detections:
        st.success("✅ No damage or accident detected!")
        return

    accident_dets = [d for d in detections if d.category == "accident"]
    damage_dets   = [d for d in detections if d.category == "damage"]

    if accident_dets:
        st.markdown("""
        <div class='accident-banner'>
            <h2>🚨 ACCIDENT / INCIDENT DETECTED</h2>
            <p>Emergency services may need to be alerted immediately.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("**Detected Objects:**")
    for d in detections:
        sev_bg, sev_fg, sev_icon = SEVERITY_COLOURS.get(
            d.severity, ("#f1f5f9", "#334155", "⬜"))
        dot_css = _colour_hex_to_bgr_css(d.color)
        st.markdown(f"""
        <div class='det-item'>
            <div class='det-dot' style='background:{dot_css}'></div>
            <div>
                <div class='det-name'>{d.label}</div>
                <div class='det-conf'>Confidence: {d.confidence:.0%}  |  Category: {d.category.title()}</div>
            </div>
            <span class='det-sev' style='background:{sev_bg};color:{sev_fg}'>
                {sev_icon} {d.severity}
            </span>
        </div>""", unsafe_allow_html=True)


def _render_damage_gauge(score: float, condition: str, severity: str):
    cond_clr   = CONDITION_COLOURS.get(condition, "#64748b")
    sev_bg, sev_fg, sev_icon = SEVERITY_COLOURS.get(severity, ("#f1f5f9","#334155","⬜"))
    # Bar colour
    if score < 20:   bar_clr = "#10b981"
    elif score < 45: bar_clr = "#f59e0b"
    elif score < 65: bar_clr = "#f97316"
    else:            bar_clr = "#ef4444"

    st.markdown(f"""
    <div class='gauge-card'>
        <div class='gauge-lbl'>Road Damage Score</div>
        <div class='gauge-val' style='color:{bar_clr}'>{score:.0f}</div>
        <div style='font-size:0.72rem;color:#94a3b8'>out of 100</div>
        <div class='damage-bar-wrap' style='margin:12px 0'>
            <div class='damage-bar-fill' style='width:{score}%;background:{bar_clr}'></div>
        </div>
        <div class='gauge-cond' style='color:{cond_clr}'>{condition} Condition</div>
        <span class='det-sev' style='background:{sev_bg};color:{sev_fg}'>
            {sev_icon} {severity} Severity
        </span>
    </div>""", unsafe_allow_html=True)


def _image_detection_tab():
    st.markdown("""
    <div class='upload-zone'>
        <h3>📸 Upload Road Image</h3>
        <p>Supports JPG, PNG, JPEG — AI will detect potholes, cracks, road damage & accidents</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
        label_visibility="collapsed",
    )

    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        add_to_db = st.checkbox("📥 Auto-save findings to database", value=True)
    with col_opts2:
        show_raw = st.checkbox("👁️ Show original alongside annotated", value=True)

    if uploaded is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;opacity:0.4'>
            <div style='font-size:4rem'>🛣️</div>
            <div style='font-size:0.95rem;color:#64748b;margin-top:8px'>
                Upload an image above to start analysis
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # Read image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("❌ Could not read image. Please upload a valid image file.")
        return

    with st.spinner("🔍 Running AI analysis…"):
        time.sleep(0.6)   # simulate inference time
        result = mock_analyze_image(img_bgr)

    # ── Layout ─────────────────────────────────────────────────────────────────
    col_img, col_results = st.columns([3, 2])

    with col_img:
        if show_raw:
            tab_ann, tab_orig = st.tabs(["🤖 AI Annotated", "📷 Original"])
            with tab_ann:
                ann_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True, caption="AI Detection Output")
            with tab_orig:
                orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True, caption="Original Upload")
        else:
            ann_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True, caption="AI Detection Output")

        # Download annotated image
        pil_img   = Image.fromarray(ann_rgb)
        buf       = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        st.download_button(
            "⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name="vigiroad_annotated.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    with col_results:
        _render_damage_gauge(
            result["damage_score"],
            result["road_condition"],
            result["severity"],
        )

        st.markdown("---")
        _render_image_detections(result["detections"])

        st.markdown("---")
        st.markdown("**📝 Analysis Summary:**")
        for line in result["summary"]:
            st.markdown(f"<div style='font-size:0.86rem;color:#334155;margin:4px 0'>{line}</div>",
                        unsafe_allow_html=True)

        # Save to DB
        if add_to_db and result["detections"]:
            loc = get_random_tn_location()
            for d in result["detections"]:
                rtype = "accident" if d.category == "accident" else "pothole"
                add_report(rtype, d.severity.lower(),
                           loc["name"], loc["lat"], loc["lon"])
            st.success(f"✅ Saved {len(result['detections'])} finding(s) to database.")

        if result["accident_detected"]:
            st.error("⚠️ Accident detected – check dashboards for alerts.")


# ─────────────────────────────────────────────────────────────────────────────
# Section: Video Detection
# ─────────────────────────────────────────────────────────────────────────────

def _render_video_summary_cards(summary: dict):
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "📹", "Total Frames", str(summary["total_frames"]), "#6366f1"),
        (c2, "🕳️", "Damaged Frames", str(summary["damaged_frame_count"]), "#f59e0b"),
        (c3, "🚨", "Accident Frames", str(summary["accident_frame_count"]), "#ef4444"),
        (c4, "📊", "Avg Damage Score", f"{summary['avg_damage_score']}/100", "#0ea5e9"),
    ]
    for col, icon, label, val, clr in cards:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='border-top:3px solid {clr};text-align:center'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div class='value' style='color:{clr};font-size:1.6rem'>{val}</div>
                <div class='label'>{label}</div>
            </div>""", unsafe_allow_html=True)


def _render_damage_timeline(frame_results):
    """Render a compact damage score timeline as an SVG sparkline."""
    if not frame_results:
        return

    scores = [f.damage_score for f in frame_results]
    n      = len(scores)
    W, H   = 600, 80
    pad    = 5

    # Points
    pts = []
    for i, s in enumerate(scores):
        x = pad + int((i / max(n - 1, 1)) * (W - 2 * pad))
        y = pad + int(((100 - s) / 100) * (H - 2 * pad))
        pts.append((x, y))

    # Build SVG path
    path = "M " + " L ".join(f"{x},{y}" for x, y in pts)
    fill_pts = [(pad, H - pad)] + pts + [(W - pad, H - pad)]
    fill_path = "M " + " L ".join(f"{x},{y}" for x, y in fill_pts) + " Z"

    accident_lines = ""
    for fr in frame_results:
        if fr.accident_detected and n > 1:
            ax = pad + int((fr.frame_index / max(n - 1, 1)) * (W - 2 * pad))
            accident_lines += f'<line x1="{ax}" y1="{pad}" x2="{ax}" y2="{H-pad}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,2"/>'

    svg = f"""
    <svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
         style="display:block;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0;margin:8px 0">
        <defs>
            <linearGradient id="dmgGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#6366f1" stop-opacity="0.35"/>
                <stop offset="100%" stop-color="#6366f1" stop-opacity="0.02"/>
            </linearGradient>
        </defs>
        <path d="{fill_path}" fill="url(#dmgGrad)"/>
        <path d="{path}" stroke="#6366f1" stroke-width="2" fill="none" stroke-linejoin="round"/>
        {accident_lines}
        <text x="8" y="14" fill="#94a3b8" font-size="9" font-family="Inter,sans-serif">100</text>
        <text x="8" y="{H-4}" fill="#94a3b8" font-size="9" font-family="Inter,sans-serif">0</text>
        <text x="{W//2-40}" y="{H-2}" fill="#94a3b8" font-size="9" font-family="Inter,sans-serif">Damage Score Timeline</text>
    </svg>"""
    st.markdown(svg, unsafe_allow_html=True)
    if any(f.accident_detected for f in frame_results):
        st.markdown("<div style='font-size:0.78rem;color:#ef4444'>🔴 Red vertical lines = accident detected frames</div>",
                    unsafe_allow_html=True)


def _video_detection_tab():
    st.markdown("""
    <div class='upload-zone'>
        <h3>🎬 Upload Road Video</h3>
        <p>Supports MP4, AVI, MOV — AI analyses every Nth frame for road damage & accident events</p>
    </div>""", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Choose a road video",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        key="vid_uploader",
        label_visibility="collapsed",
    )

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        frame_skip = st.slider("Analyse every N frames", 5, 30, 10,
                                help="Lower = more thorough but slower")
    with col_o2:
        max_frames = st.slider("Max frames to process", 50, 500, 150)
    with col_o3:
        vid_db = st.checkbox("📥 Save findings to DB", value=True)

    if uploaded_video is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;opacity:0.4'>
            <div style='font-size:4rem'>🎥</div>
            <div style='font-size:0.95rem;color:#64748b;margin-top:8px'>
                Upload a video above to start analysis
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # Write to temp file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    tfile.close()

    try:
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("❌ Cannot open video. Please try a different file.")
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_vid  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.markdown(f"""
        <div style='display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 16px'>
            <span class='stat-pill'>🎬 {total_vid} total frames</span>
            <span class='stat-pill'>⚡ {fps:.0f} FPS</span>
            <span class='stat-pill'>⏱ {total_vid/fps:.1f}s duration</span>
            <span class='stat-pill'>🔍 Sampling every {frame_skip} frames</span>
        </div>""", unsafe_allow_html=True)

        run_btn = st.button("▶️ Run AI Video Analysis", use_container_width=False,
                            type="primary")

        if not run_btn:
            # Just show preview frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, preview = cap.read()
            if ret:
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                         caption="Video preview (first frame)", use_container_width=True)
            cap.release()
            return

        # ── Run analysis ─────────────────────────────────────────────────────
        progress_bar    = st.progress(0, text="Initialising…")
        frame_preview   = st.empty()
        status_text     = st.empty()

        frame_results = []
        frame_idx     = 0
        processed     = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret or processed >= max_frames:
                break

            if frame_idx % frame_skip == 0:
                result = mock_analyze_video_frame(frame, frame_idx, fps)
                frame_results.append(result)

                annotated_f = annotate_video_frame(frame, result)
                frame_rgb   = cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB)
                frame_preview.image(frame_rgb, use_container_width=True,
                                    caption=f"Frame {frame_idx}  |  "
                                            f"Score: {result.damage_score:.0f}  |  "
                                            f"Condition: {result.road_condition}"
                                            + ("  🚨 ACCIDENT!" if result.accident_detected else ""))
                processed += 1

                pct = min(processed / max_frames, 1.0)
                progress_bar.progress(pct,
                    text=f"Analysing… {processed}/{max_frames} frames  "
                         f"| Accidents: {sum(1 for r in frame_results if r.accident_detected)}")

                if result.accident_detected:
                    status_text.error(f"🚨 Accident detected at {result.timestamp_sec:.1f}s"
                                      f" (Frame {frame_idx})")
                elif result.damage_score > 50:
                    status_text.warning(f"⚠️ High damage at {result.timestamp_sec:.1f}s"
                                        f" – Score {result.damage_score:.0f}")
                else:
                    status_text.info(f"✅ Frame {frame_idx} – "
                                     f"Condition: {result.road_condition}")

            frame_idx += 1

        cap.release()
        progress_bar.progress(1.0, text="✅ Analysis complete!")
        frame_preview.empty()
        status_text.empty()

        if not frame_results:
            st.warning("No frames were analysed.")
            return

        # ── Summary ───────────────────────────────────────────────────────────
        summary = build_video_summary(frame_results, fps)

        st.markdown("---")
        st.markdown("<div class='section-header' style='font-size:1.3rem'>📊 Video Analysis Report</div>",
                    unsafe_allow_html=True)

        _render_video_summary_cards(summary)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("**📈 Damage Score Timeline**")
            _render_damage_timeline(frame_results)

            # Accident timestamps
            if summary["accident_timestamps"]:
                st.markdown("**🚨 Accident Events:**")
                for ts in summary["accident_timestamps"]:
                    mins = int(ts // 60)
                    secs = ts % 60
                    st.markdown(f"""
                    <div class='timeline-item'>
                        🚨 <b>Accident detected</b> at
                        <code>{mins:02d}:{secs:05.2f}</code>
                        (Frame {int(ts * fps)})
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ No accidents detected in the video.")

        with col_right:
            # Overall road condition gauge
            _render_damage_gauge_mini(summary)

            st.markdown("---")
            # Damage type breakdown
            if summary["damage_type_counts"]:
                st.markdown("**🔍 Damage Type Breakdown:**")
                sorted_types = sorted(summary["damage_type_counts"].items(),
                                      key=lambda x: x[1], reverse=True)
                for label, count in sorted_types:
                    all_labels = {**DAMAGE_LABELS, **ACCIDENT_LABELS}
                    info  = all_labels.get(label, {"color": (100, 100, 100)})
                    dot   = _colour_hex_to_bgr_css(info["color"])
                    st.markdown(f"""
                    <div class='det-item'>
                        <div class='det-dot' style='background:{dot}'></div>
                        <div class='det-name'>{label}</div>
                        <span class='stat-pill' style='margin-left:auto'>{count}×</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ No damage types detected.")

        # Save to DB
        if vid_db:
            saved = 0
            loc   = get_random_tn_location()
            for fr in frame_results:
                for d in fr.detections:
                    rtype = "accident" if d.category == "accident" else "pothole"
                    add_report(rtype, d.severity.lower(),
                                loc["name"], loc["lat"], loc["lon"])
                    saved += 1
            if saved:
                st.success(f"✅ {saved} detection(s) saved to database.")

    finally:
        try:
            os.unlink(tfile.name)
        except Exception:
            pass


def _render_damage_gauge_mini(summary: dict):
    score    = summary["avg_damage_score"]
    peak     = summary["peak_damage_score"]
    cond     = summary["overall_condition"]
    sev      = summary["severity"]
    cond_clr = CONDITION_COLOURS.get(cond, "#64748b")

    if score < 20:   bar_clr = "#10b981"
    elif score < 45: bar_clr = "#f59e0b"
    elif score < 65: bar_clr = "#f97316"
    else:            bar_clr = "#ef4444"

    sev_bg, sev_fg, sev_icon = SEVERITY_COLOURS.get(sev, ("#f1f5f9","#334155","⬜"))

    st.markdown(f"""
    <div class='gauge-card'>
        <div class='gauge-lbl'>Avg Road Damage Score</div>
        <div class='gauge-val' style='color:{bar_clr}'>{score:.0f}</div>
        <div style='font-size:0.72rem;color:#94a3b8'>out of 100</div>
        <div class='damage-bar-wrap' style='margin:10px 0'>
            <div class='damage-bar-fill' style='width:{score}%;background:{bar_clr}'></div>
        </div>
        <div class='gauge-cond' style='color:{cond_clr}'>{cond} Condition</div>
        <span class='det-sev' style='background:{sev_bg};color:{sev_fg}'>{sev_icon} {sev}</span>
        <div style='margin-top:12px;font-size:0.78rem;color:#64748b'>
            Peak Score: <b style='color:#ef4444'>{peak:.0f}/100</b>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main page entry point
# ─────────────────────────────────────────────────────────────────────────────

def show_media_detection():
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>🔬 AI Media Detection</div>
    <div class='section-sub'>
        Upload an image or video for intelligent road condition analysis —
        pothole detection, damage scoring, and accident identification
    </div>""", unsafe_allow_html=True)

    tab_image, tab_video, tab_help = st.tabs([
        "📸 Image Detection",
        "🎬 Video Detection",
        "ℹ️ How It Works",
    ])

    with tab_image:
        _image_detection_tab()

    with tab_video:
        _video_detection_tab()

    with tab_help:
        st.markdown("""
        ## How VigiRoad AI Media Detection Works

        ### 📸 Image Detection
        | Step | What Happens |
        |------|-------------|
        | 1 | Upload a JPG/PNG road image |
        | 2 | AI scans for potholes, cracks, rutting, water logging, debris, road collapse, and accidents |
        | 3 | Bounding boxes and labels are drawn on the frame |
        | 4 | A 0–100 damage score is computed using weighted detection counts |
        | 5 | Road condition (Good → Critical) is mapped from the damage score |
        | 6 | Findings can be auto-saved to the VigiRoad database |

        ### 🎬 Video Detection
        | Step | What Happens |
        |------|-------------|
        | 1 | Upload an MP4/AVI/MOV video |
        | 2 | AI samples every N frames (configurable) to balance speed vs. thoroughness |
        | 3 | Each sampled frame is analysed for road damage AND accidents |
        | 4 | A timeline graph shows damage score variation across the video |
        | 5 | Accident timestamps are listed with frame numbers for review |
        | 6 | A full breakdown of damage types is presented in the report |

        ### 🎯 Detection Classes
        | Class | Type | Notes |
        |-------|------|-------|
        | Pothole | Damage | Severity based on size and count |
        | Crack | Damage | Surface fractures |
        | Rutting | Damage | Wheel path deformation |
        | Edge Break | Damage | Road edge deterioration |
        | Debris | Damage | Foreign objects on road |
        | Water Logging | Damage | Standing water / flooding |
        | Road Collapse | Damage | Critical structural failure |
        | Accident | Accident | Vehicle collision event |
        | Collision | Accident | Direct impact between vehicles |
        | Vehicle Skid | Accident | Loss of traction event |
        | Rollover | Accident | Vehicle rollover incident |

        ### 🔧 Production Upgrade
        In production, replace the mock inference functions in `utils/media_detection.py`
        with real **YOLOv8 / YOLOv9** model calls:
        ```python
        from ultralytics import YOLO
        model = YOLO("best.pt")
        results = model(frame)
        # parse results.boxes for bounding boxes, classes, confidence
        ```
        """)
